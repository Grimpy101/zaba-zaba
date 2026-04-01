import argparse
import numpy
import soundfile

import torch

import tensorflow_hub
import tensorflow
tensorflow.experimental.numpy.experimental_enable_numpy_behavior()

from ..zabe_v1 import perch
import audio_processing


OVERLAP_DURATION_SECS = 1.0
SNIPPET_DURATION_SECS  = 5.0


def pytorch_perch_inference(
    snippets,
    perch_path: str
):
    perchv2 = perch.PerchV2()
    perchv2.load_from_onnx(perch_path)
    
    snippets = torch.from_numpy(snippets)
    
    try:
        with torch.no_grad():
            embeddings, _, _, _ = perchv2(snippets)
            return embeddings.numpy(force=True)
    except Exception as e:
        print(f"Failed: {e}")
    return None


def tensorflow_perch_inference(snippets):
    model = tensorflow_hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2')
    model_outputs = model.signatures['serving_default'](inputs=snippets[numpy.newaxis, :])
    embeddings = model_outputs['embedding']
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument('--audio')
    _ = parser.add_argument('--perchv2', required=True)
    _ = parser.add_argument('--synthetic', action='store_true')
    
    arguments = parser.parse_args()
    
    (snippets, sample_rate) = audio_processing.preprocess_file(arguments.audio, 5.0, 1.0)
    embeddings_torch = pytorch_perch_inference(snippets, arguments.perchv2)
    embeddings_tensorflow = tensorflow_perch_inference(snippets)
    
    print(embeddings_torch)
    print(embeddings_tensorflow)
    
    if embeddings_torch == embeddings_tensorflow:
        print("Enako!")