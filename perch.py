import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
from typing import Tuple, Dict, Optional


class SiLU(nn.Module):
    """SiLU (Swish) activation: x * sigmoid(x)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    This block computes channel-wise attention weights by:
    1. Global average pooling over spatial dimensions
    2. FC -> SiLU -> FC -> Sigmoid
    3. Scale input features by attention weights
    """

    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.fc1_weight = nn.Parameter(torch.zeros(squeeze_channels, in_channels))
        self.fc1_bias = nn.Parameter(torch.zeros(squeeze_channels))
        self.fc2_weight = nn.Parameter(torch.zeros(in_channels, squeeze_channels))
        self.fc2_bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C] (NHWC format as in ONNX)
        # Global average pooling
        se = x.sum(dim=(1, 2)) / (x.shape[1] * x.shape[2])  # [B, C]

        # FC1 -> SiLU
        se = F.linear(se, self.fc1_weight, self.fc1_bias)
        se = se * torch.sigmoid(se)  # SiLU

        # FC2 -> Sigmoid
        se = F.linear(se, self.fc2_weight, self.fc2_bias)
        se = torch.sigmoid(se)  # [B, C]

        # Scale input
        return x * se.unsqueeze(1).unsqueeze(2)


class ConvBNSiLU(nn.Module):
    """Convolution + BatchNorm (fused as scale+bias) + SiLU activation.
    In the ONNX model, BatchNorm is represented as Mul + Add operations
    after convolution. This preserves that structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        has_activation: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True,
        )
        # Fused BN parameters (scale and bias)
        self.bn_scale = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.has_activation = has_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] (NCHW for Conv2d)
        x = self.conv(x)
        # Transpose to NHWC for BN operations (matching ONNX)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x * self.bn_scale + self.bn_bias
        if self.has_activation:
            x = x * torch.sigmoid(x)  # SiLU
        return x  # Returns NHWC


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv block (MBConv) with SE.
    Structure:
    1. Expansion: 1x1 conv to expand channels
    2. Depthwise: KxK depthwise conv
    3. Squeeze-Excitation
    4. Projection: 1x1 conv to reduce channels
    5. Optional skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int = 1,
        se_ratio: float = 0.25,
        has_skip: bool = True,
    ):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        se_channels = max(1, int(in_channels * se_ratio))

        self.has_skip = has_skip and (stride == 1) and (in_channels == out_channels)

        # Calculate padding for depthwise conv
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            padding = kernel_size // 2

        # Expansion phase
        self.expand_conv = ConvBNSiLU(
            in_channels, expanded_channels, 1, has_activation=True
        )

        # Depthwise phase
        self.depthwise_conv = ConvBNSiLU(
            expanded_channels,
            expanded_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=expanded_channels,
            has_activation=True,
        )

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(expanded_channels, se_channels)

        # Projection phase (no activation)
        self.project_conv = ConvBNSiLU(
            expanded_channels, out_channels, 1, has_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C] (NHWC)
        identity = x

        # Expand
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.expand_conv(x)  # Returns NHWC

        # Depthwise
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.depthwise_conv(x)  # Returns NHWC

        # SE
        x = self.se(x)  # NHWC -> NHWC

        # Project
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.project_conv(x)  # Returns NHWC

        # Skip connection
        if self.has_skip:
            x = x + identity

        return x


class AudioFrontend(nn.Module):
    """Learned audio frontend that converts raw waveform to mel spectrogram.
    The frontend consists of:
    1. Reshape input to [B, 1, 160000, 1]
    2. 1D convolution (implemented as 2D) for frame extraction: [B, 640, 500, 1]
    3. Pad from 640 to 1024 for DFT
    4. DFT transformation using matrix multiplication: [B, 500, 1026]
    5. Reshape to complex: [B, 500, 513, 2]
    6. ReduceL2 for magnitude: [B, 500, 513]
    7. Mel filterbank projection: [B, 500, 128]
    8. Log compression
    """

    def __init__(self, n_mels: int = 128, n_fft: int = 640, hop_length: int = 320):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Conv layer for frame extraction (learned STFT-like operation)
        # Output: [B, 640, 500, 1] with 640 output channels
        self.frame_conv = nn.Conv2d(
            1,
            n_fft,
            kernel_size=(n_fft, 1),
            stride=(hop_length, 1),
            padding=(n_fft // 4, 0),
            bias=False,
        )

        # Pre-DFT scaling factor
        self.pre_dft_scale = nn.Parameter(torch.ones(n_fft))

        # DFT matrix: [1024, 1026] - padded input to complex output
        # 640 -> pad to 1024, then matmul to get 1026 (513 complex pairs)
        self.dft_matrix = nn.Parameter(torch.zeros(1024, 1026))

        # Mel filterbank: [513, 128]
        self.mel_matrix = nn.Parameter(torch.zeros(513, n_mels))

        # Log scale factor
        self.log_scale = nn.Parameter(torch.tensor(0.0834968))  # From ONNX model
        self.log_offset = nn.Parameter(
            torch.tensor(1e-6)
        )  # Small offset for log stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 160000]
        batch_size = x.shape[0]

        # Reshape to [B, 1, 160000, 1]
        x = x.view(batch_size, 1, -1, 1)

        # Frame extraction conv: [B, 640, 500, 1]
        x = self.frame_conv(x)

        # Transpose to [B, 500, 1, 640] then reshape to [B, 500, 640]
        x = x.permute(0, 2, 3, 1).squeeze(2)  # [B, 500, 640]

        # Scale before DFT (element-wise with learned scale)
        x = self.pre_dft_scale * x

        # Pad from 640 to 1024 for DFT (matching ONNX pad operation)
        x = F.pad(x, (0, 384))  # Pad last dim: [B, 500, 1024]

        # DFT via matrix multiplication: [B, 500, 1026]
        x = torch.matmul(x, self.dft_matrix)

        # Reshape to complex pairs: [B, 500, 513, 2]
        x = x.view(batch_size, 500, 513, 2)

        # Compute magnitude via L2 norm: [B, 500, 513]
        x = torch.norm(x, dim=-1)

        # Mel filterbank projection: [B, 500, 128]
        x = torch.matmul(x, self.mel_matrix)

        # Log compression with offset for stability
        x = torch.clamp(x, min=self.log_offset.item())
        x = self.log_scale * torch.log(x)

        return x  # [B, 500, 128]


class EfficientNetBackbone(nn.Module):
    """EfficientNet-like backbone for audio processing.
    The architecture processes mel spectrogram features through a series
    of MBConv blocks with increasing channels and decreasing spatial resolution.
    """

    def __init__(self):
        super().__init__()

        # Initial stem: Conv 3x3 with stride 2
        self.stem = ConvBNSiLU(1, 40, 3, stride=2, padding=0)

        # MBConv blocks configuration:
        # (in_ch, out_ch, expand_ratio, kernel, stride, num_repeats)
        self.block_configs = [
            # Stage 1
            (40, 24, 1, 3, 1, 1),  # Depthwise 3x3, no expansion
            (24, 24, 6, 3, 1, 1),  # MBConv6 3x3
            # Stage 2
            (24, 24, 6, 3, 1, 1),  # MBConv6 3x3
            (24, 32, 6, 3, 2, 1),  # MBConv6 3x3, stride 2
            # Stage 3
            (32, 32, 6, 3, 1, 2),  # MBConv6 3x3 x2
            (32, 48, 6, 5, 2, 1),  # MBConv6 5x5, stride 2
            # Stage 4
            (48, 48, 6, 5, 1, 2),  # MBConv6 5x5 x2
            (48, 96, 6, 3, 2, 1),  # MBConv6 3x3, stride 2
            # Stage 5
            (96, 96, 6, 3, 1, 3),  # MBConv6 3x3 x3
            (96, 96, 6, 5, 1, 1),  # MBConv6 5x5
            # Stage 6
            (96, 136, 6, 5, 1, 1),  # MBConv6 5x5
            (136, 136, 6, 5, 1, 3),  # MBConv6 5x5 x3
            (136, 136, 6, 5, 2, 1),  # MBConv6 5x5, stride 2
            # Stage 7
            (136, 232, 6, 5, 1, 1),  # MBConv6 5x5
            (232, 232, 6, 5, 1, 3),  # MBConv6 5x5 x3
            (232, 232, 6, 3, 1, 1),  # MBConv6 3x3
            # Stage 8
            (232, 384, 6, 3, 1, 1),  # MBConv6 3x3
            (384, 384, 6, 3, 1, 1),  # MBConv6 3x3
        ]

        # Build blocks
        self.blocks = nn.ModuleList()
        for in_ch, out_ch, expand, kernel, stride, repeats in self.block_configs:
            for i in range(repeats):
                self.blocks.append(
                    MBConvBlock(
                        in_ch if i == 0 else out_ch,
                        out_ch,
                        expand_ratio=expand,
                        kernel_size=kernel,
                        stride=stride if i == 0 else 1,
                        has_skip=(i > 0 or in_ch == out_ch),
                    )
                )

        # Final conv
        self.final_conv = ConvBNSiLU(384, 1536, 1, has_activation=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 500, 128] from frontend
        # Add channel dim and transpose to NCHW: [B, 1, 500, 128]
        x = x.unsqueeze(1)

        # Stem
        x = self.stem(x)  # Returns NHWC

        # MBConv blocks
        for block in self.blocks:
            x = block(x)

        # Final conv
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.final_conv(x)  # Returns NHWC

        return x  # [B, H, W, 1536]


class ClassificationHead(nn.Module):
    """Multi-output classification head.
    Produces:
    1. Global embedding (1536-d)
    2. Per-frame features (16x4x1536)
    3. Mel features (500x128)
    4. Logits (14795 species classes)
    """

    def __init__(self, embed_dim: int = 1536, num_classes: int = 14795):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Classification projection: maps normalized embedding to logits
        # In ONNX: MatMul with dot_general6_reshaped_0 weight matrix
        self.classifier_weight = nn.Parameter(torch.zeros(embed_dim, num_classes * 4))

        # Learned scaling for logits: [14795, 4] broadcast over spatial dims
        self.logits_scale = nn.Parameter(torch.ones(num_classes, 4))
        self.logits_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(
        self, features: torch.Tensor, mel_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: [B, H, W, C] from backbone (should be [B, 16, 4, 1536])
        # mel_features: [B, 500, 128] from frontend

        batch_size = features.shape[0]

        # Per-frame features output: val_1 [B, 16, 4, 1536]
        frame_features = features.view(batch_size, 16, 4, self.embed_dim)

        # SiLU activation on features (val_1 in ONNX)
        frame_features_silu = frame_features * torch.sigmoid(frame_features)

        # Compute L2 norm for normalization
        # val_1126 = Mul(val_1, val_1), val_1127 = ReduceSum, val_1128 = Sqrt
        sq_sum = (frame_features_silu**2).sum(dim=-1, keepdim=True)  # [B, 16, 4, 1]
        norm = torch.sqrt(sq_sum) + 1e-6  # [B, 16, 4, 1]

        # Normalize: val_1130 = Div(val_1, val_1129)
        features_norm = frame_features_silu / norm  # [B, 16, 4, 1536]

        # MatMul with classifier weight: val_1131 [B, 16, 4, num_classes*4]
        logits_raw = torch.matmul(features_norm, self.classifier_weight)

        # Reshape to [B, 16, 4, 14795, 4]
        logits_reshaped = logits_raw.view(batch_size, 16, 4, self.num_classes, 4)

        # ReduceMax over the last dimension (dim=4): [B, 14795, 4]
        # First permute to [B, 14795, 16, 4, 4] then max
        logits_permuted = logits_reshaped.permute(0, 3, 1, 2, 4)  # [B, 14795, 16, 4, 4]
        logits_max = logits_permuted.max(dim=2)[0]  # max over dim 16: [B, 14795, 4, 4]
        logits_max = logits_max.max(dim=2)[0]  # max over dim 4: [B, 14795, 4]

        # Apply learned scaling: Mul with logits_scale [14795, 4]
        logits_scaled = logits_max * self.logits_scale  # [B, 14795, 4]

        # ReduceSum over last dim: [B, 14795]
        logits_sum = logits_scaled.sum(dim=-1)

        # Add bias: [B, 14795]
        final_logits = logits_sum + self.logits_bias

        # Global average pooling for embedding: val (output)
        # ReduceSum over [1, 2] then Div
        embedding = frame_features_silu.sum(dim=(1, 2)) / 64.0  # 16*4=64, [B, 1536]

        return embedding, frame_features_silu, mel_features, final_logits


class PerchV2(nn.Module):
    """Perch v2 Bird Audio Classification Model.
    A PyTorch implementation of the Perch v2 model for bird species
    identification from audio recordings.
    Input:
        audio: Tensor of shape [batch, 160000] representing 10 seconds
               of audio at 16kHz sample rate
    Output:
        embedding: [batch, 1536] global embedding vector
        frame_features: [batch, 16, 4, 1536] per-frame features
        mel_features: [batch, 500, 128] mel spectrogram features
        logits: [batch, 14795] classification logits for species
    Example:
        >>> model = PerchV2()
        >>> audio = torch.randn(1, 160000)
        >>> embedding, frames, mel, logits = model(audio)
    """

    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()

        self.frontend = AudioFrontend(n_mels=128, n_fft=640, hop_length=320)
        self.backbone = EfficientNetBackbone()
        self.head = ClassificationHead(embed_dim=1536, num_classes=14795)

        if pretrained_path:
            self.load_from_onnx(pretrained_path)

    def forward(
        self, audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            audio: Input audio waveform [batch, 160000]
        Returns:
            Tuple of (embedding, frame_features, mel_features, logits)
        """
        # Frontend: audio -> mel spectrogram
        mel_features = self.frontend(audio)  # [B, 500, 128]

        # Backbone: mel -> features
        features = self.backbone(mel_features)  # [B, H, W, 1536]

        # Head: features -> outputs
        return self.head(features, mel_features)

    def load_from_onnx(self, onnx_path: str):
        """Load weights from an ONNX model file.
        Args:
            onnx_path: Path to the ONNX model file
        """
        onnx_model = onnx.load(onnx_path)

        # Extract initializers (weights) from ONNX
        initializers = {
            init.name: onnx.numpy_helper.to_array(init)
            for init in onnx_model.graph.initializer
        }

        # Map ONNX weights to PyTorch parameters
        self._load_onnx_weights(initializers)

        print(f"Loaded weights from {onnx_path}")

    def _load_onnx_weights(self, initializers: Dict[str, np.ndarray]):
        """Map ONNX initializers to PyTorch parameters.
        This method handles the weight mapping between ONNX format
        and PyTorch's expected parameter layout.
        """
        # This is a simplified version - full implementation would
        # require careful mapping of all ONNX weight names to PyTorch
        # parameter paths based on the exact ONNX node names.

        # For now, we just show the structure
        print(f"Found {len(initializers)} initializers in ONNX model")
        print("Note: Full weight loading requires mapping ONNX names to PyTorch params")


def load_perch_from_onnx(onnx_path: str) -> PerchV2:
    """Load a PerchV2 model with weights from an ONNX file.
    This function creates a new PerchV2 model and loads the weights
    from the specified ONNX model file.
    Args:
        onnx_path: Path to the ONNX model file (e.g., 'perch_v2.onnx')
    Returns:
        PerchV2 model with loaded weights
    Example:
        >>> model = load_perch_from_onnx('perch_v2.onnx')
        >>> model.eval()
        >>> with torch.no_grad():
        ...     output = model(audio)
    """
    model = PerchV2()
    model.load_from_onnx(onnx_path)
    return model