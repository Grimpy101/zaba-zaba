import onnxruntime
import torch
import torch.nn as nn


class PerchV2Torch(nn.Module):
    """PyTorch-compatible wrapper for Perch v2 using ONNX Runtime.

    This allows using the model in PyTorch pipelines (DataLoader,
    mixed-precision, etc.) while preserving exact numerical equivalence
    with the original TensorFlow model.
    """

    def __init__(self, onnx_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        provider = "CUDAExecutionProvider" if "cuda" in device else "CPUExecutionProvider"
        self.session = onnxruntime.InferenceSession(onnx_path, providers=[provider, "CPUExecutionProvider"])
        # Register a dummy parameter so .to() / .device work
        self.register_buffer("_dummy", torch.zeros(1))  # type: ignore

    def forward(self, audio: torch.Tensor):
        """
        Args:
            audio: [batch, 160000] float32 tensor

        Returns:
            dict with keys: embedding, logits, spatial_embedding, spectrogram
        """
        audio_np = audio.detach().cpu().numpy()
        outputs = self.session.run(None, {"inputs": audio_np})
        output_names = [o.name for o in self.session.get_outputs()]
        result = {}
        for name, val in zip(output_names, outputs):
            key = "logits" if name == "label" else name
            result[key] = torch.from_numpy(val).to(audio.device)  # type: ignore
        return result