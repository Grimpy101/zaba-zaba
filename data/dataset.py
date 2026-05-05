import os
import typing

import torch
import numpy

import soundfile
import tqdm

from ..utils import audio_processing


SPECIES = {
    'Hyla_arborea': 0,
    'Pelophylax': 1,
    'Bufo_viridis': 2,
    'Other': 3
}


class ZabeDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, data_config: dict):
        self.sample_rate = data_config['sample_rate']
        self.sample_length = data_config['sample_length']
        self.sample_overlap = data_config['sample_overlap']
        self.data_root = data_config['data_root']
        self.unique_labels = set()
        
        species_folders = [
            item
            for item in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, item))
        ]
        print(f"{len(species_folders)} species")
        
        step = self.sample_length - self.sample_overlap
        
        labels = []
        files = []
        indices = []
        
        for species_folder in tqdm.tqdm(species_folders):
            label_id = self.species_label(species_folder)
            self.unique_labels.add(label_id)
            
            full_path = os.path.join(self.data_root, species_folder)
            filepaths = [
                os.path.join(full_path, item)
                for item in os.listdir(full_path)
                if os.path.isfile(os.path.join(full_path, item))
            ]
            
            for filepath in filepaths:
                info = soundfile.info(filepath)
                n_frames = info.frames
                a = numpy.arange(0, n_frames, step)
                n: int = a.shape[0]
                for i in range(n):
                    files.append(filepath)
                    indices.append(i)
                    labels.append(label_id)
        
        self.labels = numpy.array(labels)
        self.files = numpy.array(files)
        self.indices = numpy.array(indices)
    
    @staticmethod
    def species_label(species: str):
        label_text = 'Other'
        if 'Pelophylax' in species:
            label_text = 'Pelophylax'
        elif 'Hyla_arborea' == species:
            label_text = 'Hyla_arborea'
        elif 'Bufotes_viridis' == species:
            label_text = 'Bufo_viridis'
        return SPECIES.get(label_text, 3)
    
    def __len__(self):
        return len(self.indices)
    
    @property
    def classes(self) -> int:
        return len(self.unique_labels)
    
    @typing.override
    def __getitem__(self, index: int) -> tuple[numpy.ndarray, int]:  # pyright: ignore[reportIncompatibleMethodOverride]
        file = self.files[index]
        i: int = self.indices[index]
        label = self.labels[index]
        audio, _ = audio_processing.preprocess_file(file, self.sample_length, self.sample_overlap)
        audio = audio[i]
        return audio, label