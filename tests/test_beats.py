
import argparse
import torch
from beats import BEATs
from utils import audio_processing
import soundfile
import numpy


def main(arguments):
    audio_filepath = arguments.audio
    model_file = arguments.beats_model
    
    checkpoint = torch.load(model_file)
    config = BEATs.BEATsConfig(checkpoint['cfg'])
    model = BEATs.BEATs(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    audio, sample_rate = soundfile.read(audio_filepath, always_2d=False, dtype='float32')
    audio = numpy.expand_dims(audio, axis=0)
    audio = torch.from_numpy(audio)
    padding_mask = torch.zeros(audio.shape).bool()
    
    print(audio.shape)
    representation = model.extract_features(audio, padding_mask=padding_mask)[0]
    embeddings = representation.mean(dim=1)
    print(embeddings.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument('--audio')
    _ = parser.add_argument('--beats_model', required=True)
    
    arguments = parser.parse_args()
    main(arguments)