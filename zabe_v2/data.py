import collections
import csv
import dataclasses
import math
import os
import typing

import numpy
import numpy.typing
import sklearn
import sklearn.model_selection
import torch

import tqdm
import soundfile
import librosa


class ZabeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, sample_length: int, overlap: int):
        # Length of each audio sample
        self.sample_length = sample_length
        self.classes = 0

        species_folders = [
            item
            for item in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, item))
        ]
        
        print(f"{len(species_folders)} species")
        
        final_files = []
        audio_offsets = []
        labels = []
        
        with open("labels.csv", 'w') as f:
            writer = csv.writer(f)
            for i, species in enumerate(species_folders):
                writer.writerow((str(i), species))
        
        i = 0
        for species_folder in tqdm.tqdm(species_folders):
            full_path = os.path.join(data_dir, species_folder)
            files = [os.path.join(full_path, item) for item in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, item))]
            
            audio_lengths = []
            
            duration = 0.0
            for file in files:
                with soundfile.SoundFile(file, mode='r') as f:
                    duration += (f.frames / f.samplerate)
                    audio_lengths.append(f.frames)
            
            if duration < 50.0:
                continue
            
            for (file, audio_length) in zip(files, audio_lengths):
                step = sample_length - overlap
                offsets = numpy.arange(0, audio_length - sample_length + 1, step)
                for offset in offsets:
                    final_files.append(file)
                    audio_offsets.append(offset)
                    labels.append(i)
            i += 1
        
        # Three arrays combined represent a sample at each index
        #  files - contains filepath to audio file
        self.files = final_files
        #  offsets - contains starting index of the sampled audio in the audio file
        self.audio_offsets = numpy.array(audio_offsets)
        #  labels - the classification of the sample
        self.labels = numpy.array(labels)
        
        self.unique = numpy.unique(self.labels).shape[0]
    
    def __len__(self):
        return len(self.audio_offsets)

    def __getitem__(self, index: int)  -> typing.Tuple[torch.Tensor, int]:
        # We only have the file we need to read, the offset into the file, and the label
        file = self.files[index]
        offset = self.audio_offsets[index]
        label = self.labels[index]
        
        offset = offset / 32_000
        duration = self.sample_length / 32_000
        data, _ = librosa.load(file, sr=32_000, mono=True, offset=offset, duration=duration)
        pad_width = self.sample_length - len(data)
        if pad_width > 0:
            data = numpy.pad(data, (0, pad_width), mode='constant', constant_values=0)
        data = torch.from_numpy(data)
        
        return data, label