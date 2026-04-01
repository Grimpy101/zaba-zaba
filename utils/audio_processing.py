import numpy
import soundfile


def preprocess_file(
    audio_file: str,
    snippet_duration_secs: float,
    overlap_duration_secs: float
):
    audio, sample_rate = soundfile.read(audio_file, always_2d=False)
    snippet_samples = int(snippet_duration_secs * sample_rate)
    overlap_samples = int(overlap_duration_secs * sample_rate)
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
    
    return snippets, sample_rate