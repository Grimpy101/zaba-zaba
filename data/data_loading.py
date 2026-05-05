import collections
import math

import numpy
import sklearn
import torch

from utils import audio_processing

from . import dataset


class Data:
    def __init__(self, data_config: dict, hyperparameters: dict) -> None:
        self.augmentations: bool = hyperparameters['augmentations']
        self.mixing: bool = hyperparameters['mixing']
        self.mixing_alpha: float = hyperparameters['mixing_distribution_alpha']
        
        self.dataset = dataset.ZabeDataset(data_config)
        
        file_to_indices: dict[str, list[int]] = collections.defaultdict(list)
        for index, file_path in enumerate(self.dataset.files):
            file_to_indices[file_path].append(index)
        unique_files = list(file_to_indices.keys())
        
        file_labels = numpy.array([
            collections.Counter(self.dataset.labels[file_to_indices[f]]).most_common(1)[0][0]
            for f in unique_files
        ])
        
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
        
        self.train_dataset = torch.utils.data.Subset(
            self.dataset,
            train_indices  # pyright: ignore[reportArgumentType]
        )
        self.test_dataset = torch.utils.data.Subset(
            self.dataset,
            test_indices  # pyright: ignore[reportArgumentType]
        )
        self.validation_dataset = torch.utils.data.Subset(
            self.dataset,
            validation_indices  # pyright: ignore[reportArgumentType]
        )
        
        # We want undersampling - we weight the samples so those from larger classes are less likely to be sampled
        train_label_counts = collections.Counter(self.dataset.labels[self.train_dataset.indices])
        class_weights = {
            label: 1.0 / math.sqrt(count)
            for label, count
            in train_label_counts.items()
        }
        sample_weights = torch.tensor(  # type: ignore
            [class_weights[label]
                    for label
                    in self.dataset.labels[self.train_dataset.indices]],
            dtype=torch.double  # type: ignore
        )
        training_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,  # pyright: ignore[reportArgumentType]
            num_samples=len(sample_weights),
            replacement=True
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=hyperparameters['batch_size'],
            num_workers=4,
            sampler=training_sampler,
            pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=hyperparameters['batch_size'],
            num_workers=4,
            pin_memory=True
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=hyperparameters['batch_size'],
            num_workers=4,
            pin_memory=True
        )
    
    def process_samples(self, inputs: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mixing:
            inputs, outputs = audio_processing.combine_samples(inputs, outputs, self.mixing_alpha)
        if self.augmentations:
            inputs = audio_processing.augment(inputs, self.dataset.sample_rate)
        return inputs, outputs
        
    def print_info(self):
        print("---")
        print(f"  Train: {len(self.train_dataset)}\n  Test: {len(self.test_dataset)}\n  Validation: {len(self.validation_dataset)}")
        print(f"  {self.dataset.classes} unique classes ({self.dataset.unique_labels})")
        print("---")