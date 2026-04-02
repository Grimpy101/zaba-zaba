import numpy
import numpy.typing
import soundfile
import torch

import audiomentations
import data_mixup  # pyright: ignore[reportImplicitRelativeImport]


AUGMENTATIONS = [
    audiomentations.AddGaussianNoise(),
    audiomentations.AddGaussianSNR(),
    audiomentations.BandPassFilter(),
    audiomentations.BandStopFilter(),
    audiomentations.Gain(),
    audiomentations.HighPassFilter(),
    audiomentations.Limiter(),
    audiomentations.LowPassFilter(),
    audiomentations.PitchShift(),
    audiomentations.SevenBandParametricEQ()
]


def preprocess_file(
    audio_file: str,
    snippet_samples: int,
    overlap_samples: int
):
    audio, sample_rate = soundfile.read(audio_file, always_2d=False)
    step_samples = snippet_samples - overlap_samples
    
    snippets = []
    start = 0
    
    while start < len(audio):
        end = start + snippet_samples
        snippet = audio[start:end]
        
        if len(snippet) < snippet_samples:
            pad_width = (0, snippet_samples - len(snippet))
            snippet = numpy.pad(snippet, pad_width, mode='constant')
        
        snippets.append(snippet)
        start += step_samples
    
    snippets = numpy.array(snippets)
    snippets = snippets.astype(numpy.float32)
    
    return snippets, sample_rate


def combine_samples(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha=0.4,
    num_classes=4
):
    lam = numpy.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    
    y_one_hot = torch.zeros(batch_size, num_classes, device=x.device)
    y_one_hot.scatter_(1, y.view(-1, 1), 1)
    
    y_a_one_hot = y_one_hot
    y_b_one_hot = y_one_hot[index]
    mixed_y = lam * y_a_one_hot + (1 - lam) * y_b_one_hot
    return mixed_x, mixed_y


def augment(x: torch.Tensor, sample_rate: int):
    augmentations = audiomentations.Compose(AUGMENTATIONS, shuffle=True)
    output = augmentations(x, sample_rate)  # pyright: ignore[reportArgumentType]
    output = torch.from_numpy(output)
    return output