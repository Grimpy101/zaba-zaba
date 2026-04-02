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
from ..utils import audio_processing

import tqdm


SAMPLE_LENGTH = 160_000
OVERLAP = 16_000
BATCH_SIZE = 128

SPECIES = {
    'Hyla_arborea': 0,
    'Pelophylax': 1,
    'Bufo_viridis': 2,
    'Other': 3
}

def species_label(species: str):
    label_text = 'Other'
    if 'Pelophylax' in species:
        label_text = 'Pelophylax'
    elif 'Hyla_arborea' == species:
        label_text = 'Hyla_arborea'
    elif 'Bufotes_viridis' == species:
        label_text = 'Bufo_viridis'
    return SPECIES.get(label_text, 3)


@dataclasses.dataclass
class Sample:
    file: str
    index: int


class ZabeDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, data_root: str):
        files: list[str] = []
        indices: list[int] = []
        labels: list[int] = []
        self.sample_rate = 16
        
        species_folders = [
            item
            for item in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, item))
        ]
        print(f"{len(species_folders)} species")
        
        for species_folder in tqdm.tqdm(species_folders):
            label_id = species_label(species_folder)
            
            full_path = os.path.join(data_root, species_folder)
            filepaths = [
                os.path.join(full_path, item)
                for item in os.listdir(full_path)
                if os.path.isfile(os.path.join(full_path, item))
            ]
            
            for filepath in filepaths:
                audio, sr = audio_processing.preprocess_file(filepath, SAMPLE_LENGTH, OVERLAP)
                n: int = audio.shape[0]
                for i in range(n):
                    files.append(filepath)
                    indices.append(i)
                    labels.append(label_id)
        
        self.labels: numpy.typing.NDArray[numpy.uint32] = numpy.array(labels)
        self.files = numpy.array(files)
        self.indices = numpy.array(indices)
        
    def __len__(self):
        return len(self.indices)
    
    @typing.override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file = self.files[index]
        i: int = self.indices[index]
        label = self.labels[index]
        audio = audio_processing.preprocess_file(file, SAMPLE_LENGTH, OVERLAP)[i]
        data: torch.Tensor = torch.from_numpy(audio)
        return data, label


@dataclasses.dataclass
class DatasetData:
    training_set: torch.utils.data.DataLoader[tuple[torch.Tensor, int]]
    test_set: torch.utils.data.DataLoader[tuple[torch.Tensor, int]]
    validation_set: torch.utils.data.DataLoader[tuple[torch.Tensor, int]]
    unique_labels: int
    sample_rate: int


def get_data(data_root: str) -> DatasetData:
    dataset = ZabeDataset(data_root)
    
    file_to_indices: dict[str, list[int]] = collections.defaultdict(list)
    for index, file_path in enumerate(dataset.files):
        file_to_indices[file_path].append(index)
    unique_files = list(file_to_indices.keys())
    
    file_labels = numpy.array([
        collections.Counter(dataset.labels[file_to_indices[f]]).most_common(1)[0][0]
        for f in unique_files
    ])
    
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
    
    train_file_idx, temp_file_idx = next(sss.split(unique_files, file_labels))
    temp_file_labels = file_labels[temp_file_idx]
    test_file_idx, val_file_idx = next(
        sss_val.split(temp_file_idx, temp_file_labels)
    )
    test_file_idx = temp_file_idx[test_file_idx]
    val_file_idx  = temp_file_idx[val_file_idx]
    
    train_indices = numpy.concatenate([
        file_to_indices[unique_files[i]] for i in train_file_idx
    ])
    test_indices = numpy.concatenate([
        file_to_indices[unique_files[i]] for i in test_file_idx
    ])
    validation_indices = numpy.concatenate([
        file_to_indices[unique_files[i]] for i in val_file_idx
    ])
    
    assert not (set(train_file_idx) & set(test_file_idx) & set(val_file_idx)), \
        "File leakage detected between splits!"
    
    train_dataset = torch.utils.data.Subset(
        dataset,
        train_indices  # pyright: ignore[reportArgumentType]
    )
    test_dataset = torch.utils.data.Subset(
        dataset,
        test_indices  # pyright: ignore[reportArgumentType]
    )
    validation_dataset = torch.utils.data.Subset(
        dataset,
        validation_indices  # pyright: ignore[reportArgumentType]
    )
    
    # We want undersampling - we weight the samples so those from larger classes are less likely to be sampled
    train_label_counts = collections.Counter(dataset.labels[train_dataset.indices])
    class_weights = {
        label: 1.0 / math.sqrt(count)
        for label, count
        in train_label_counts.items()
    }
    sample_weights = torch.tensor(
        [class_weights[label]
                for label
                in dataset.labels[train_dataset.indices]],
        dtype=torch.double
    )
    training_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,  # pyright: ignore[reportArgumentType]
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=training_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE
    )
    
    #unique_classes: int = numpy.unique(dataset.labels).shape[0]
    unique_classes = 4
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Validation: {len(validation_dataset)}")
    print(f"  {unique_classes} unique classes ({numpy.unique(dataset.labels)})")
    
    return DatasetData(
        train_loader,
        test_loader,
        validation_loader,
        unique_classes,
        dataset.sample_rate
    )
