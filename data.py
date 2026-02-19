import collections
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

from labels import SPECIES


@dataclasses.dataclass
class Sample:
    file: str
    offset: int
    label: int


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
        
        files = []
        audio_offsets = []
        labels = []
        
        for species_folder in tqdm.tqdm(species_folders):
            label = SPECIES[species_folder]
            full_path = os.path.join(data_dir, species_folder)
            files = [os.path.join(full_path, item) for item in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, item))]
            
            duration = 0.0
            for file in files:
                with soundfile.SoundFile(file, mode='r') as f:
                    duration += (f.frames / f.samplerate)
            
            if duration < 50.0:
                continue
            
            for file in files:
                with soundfile.SoundFile(file, mode='r') as f:
                    audio = f.read(dtype='float32')
                step = sample_length - overlap
                offsets = numpy.arange(0, len(audio) - sample_length + 1, step)
                for offset in offsets:
                    files.append(file)
                    audio_offsets.append(offset)
                    labels.append(label)
        
        # Three arrays combined represent a sample at each index
        #  files - contains filepath to audio file
        self.files = numpy.array(files)
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
        
        # We open the audio file to read samples
        with soundfile.SoundFile(file, mode='r') as f:
            audio = f.read(dtype='float32')
        
        # We take only a fixed-length window inside audio (based on given offset), and optionally pad with zeros
        data = audio[offset:offset + self.sample_length]
        pad_width = self.sample_length - len(data)
        if pad_width > 0:
            data = numpy.pad(data, (0, pad_width), mode='constant', constant_values=0)
        data = torch.from_numpy(data)
        
        return data, label
    

def load_datasets(data_dir: str, batch_size: int):
    sample_length = 160_000
    overlap = 32_000  # How much do audio samples in a single audio clip overlap
    
    dataset = ZabeDataset(data_dir, sample_length, overlap)
    
    # We define splitting of data
    # StratifiedShuffleSplit - ensures fair distribution inside classes
    sss = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )
    sss_val = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=42
    )
    train_indices, test_indices = next(sss.split(dataset.files, dataset.labels))
    test_indices, validation_indices = next(sss_val.split(dataset.files[test_indices], dataset.labels[test_indices]))
    
    train_dataset: torch.utils.data.Subset[ZabeDataset] = torch.utils.data.Subset(dataset, train_indices)
    test_dataset: torch.utils.data.Subset[ZabeDataset] = torch.utils.data.Subset(dataset, test_indices)
    validation_dataset: torch.utils.data.Subset[ZabeDataset] = torch.utils.data.Subset(dataset, validation_indices)
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Validation: {len(validation_dataset)}")
    print(f"  {dataset.unique} unique classes")
    
    # We want undersampling - we weight the samples so those from larger classes are less likely to be sampled
    train_label_counts = collections.Counter(dataset.labels[train_dataset.indices])
    class_weights = {
        label: 1.0 / math.sqrt(count)
        for label, count in train_label_counts.items()
    }
    sample_weights = torch.tensor(
        [class_weights[label] for label in dataset.labels[train_dataset.indices]],
        dtype=torch.double
    )
    training_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, # type: ignore
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=training_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, validation_loader