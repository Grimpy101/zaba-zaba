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


SAMPLE_LENGTH = 160_000
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
    elif 'Bufo_viridis' == species:
        label_text = 'Bufo_viridis'
    return SPECIES.get(label_text, 3)


class ZabeDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, data_root: str):
        self.sample_length = SAMPLE_LENGTH
        self.files: list[str] = []
        
        labels: list[int] = []
        
        species_folders = [
            item
            for item in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, item))
        ]
        print(f"{len(species_folders)} species")
        
        for species_folder in tqdm.tqdm(species_folders):
            label_id = species_label(species_folder)
            
            full_path = os.path.join(data_root, species_folder)
            files = [
                os.path.join(full_path, item)
                for item in os.listdir(full_path)
                if os.path.isfile(os.path.join(full_path, item))
            ]
            
            self.files.extend(files)
            labels.extend([label_id] * len(files))
        
        self.labels: numpy.typing.NDArray[numpy.uint32] = numpy.array(labels)
        
    def __len__(self):
        return len(self.files)
    
    @typing.override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file = self.files[index]
        label = self.labels[index]
        
        data: numpy.typing.NDArray[numpy.float32] = numpy.load(file)
        data: torch.Tensor = torch.from_numpy(data)
        return data, label


@dataclasses.dataclass
class DatasetData:
    training_set: torch.utils.data.DataLoader[tuple[torch.Tensor, int]]
    test_set: torch.utils.data.DataLoader[tuple[torch.Tensor, int]]
    validation_set: torch.utils.data.DataLoader[tuple[torch.Tensor, int]]
    unique_labels: int


def get_data(data_root: str) -> DatasetData:
    dataset = ZabeDataset(data_root)
    
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
    
    unique_classes: int = numpy.unique(dataset.labels).shape[0]
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Validation: {len(validation_dataset)}")
    print(f"  {unique_classes} unique classes")
    
    return DatasetData(
        train_loader,
        test_loader,
        validation_loader,
        unique_classes
    )
