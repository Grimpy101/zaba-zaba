import argparse
import numpy
import soundfile

import torch

import tensorflow_hub
import tensorflow
tensorflow.experimental.numpy.experimental_enable_numpy_behavior()

from zabe_v1 import perch
from utils import audio_processing


OVERLAP_DURATION_SECS = 1.0
SNIPPET_DURATION_SECS  = 5.0


def pytorch_perch_inference(
    snippets,
    perch_path: str
):
    perchv2 = perch.PerchV2()
    perchv2.load_from_onnx(perch_path)
    
    snippets = torch.from_numpy(snippets)
    
    embeddings, _, _, _ = perchv2(snippets)
    return embeddings.numpy(force=True)


def tensorflow_perch_inference(snippets):
    model = tensorflow_hub.load('https://www.kaggle.com/api/v1/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1/download')
    model_outputs = model.signatures['serving_default'](inputs=snippets)
    embeddings = model_outputs['embedding']
    return embeddings.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument('--audio')
    _ = parser.add_argument('--perchv2', required=True)
    _ = parser.add_argument('--synthetic', action='store_true')
    
    arguments = parser.parse_args()
    
    (snippets, sample_rate) = audio_processing.preprocess_file(arguments.audio, 5.0, 1.0)
    embeddings_torch = pytorch_perch_inference(snippets, arguments.perchv2)
    embeddings_tensorflow = tensorflow_perch_inference(snippets)
    
    if (embeddings_torch == embeddings_tensorflow).all():
        print("Enako!")