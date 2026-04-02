import argparse
import os

import torch
import torchmetrics
import torchmetrics.functional.classification
import tqdm
from ..utils import gpu

from beats import BEATs
from . import data, classifier
from utils import audio_processing


def evaluate(
    beats_model: BEATs.BEATs,
    my_model: classifier.NeuralNetwork1,
    data: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    classes: int,
    sample_rate: int
):
    beats_model.eval()
    
    predictions = []
    outputs = []
    
    with torch.no_grad():
        for inputs, output in tqdm.tqdm(data):
            inputs, output = audio_processing.combine_samples(inputs, output)
            inputs = audio_processing.augment(inputs, sample_rate)
            
            padding_mask = torch.zeros(inputs.shape).bool()
            representation = beats_model.extract_features(inputs, padding_mask=padding_mask)[0]
            embeddings = representation.mean(dim=1)
            prediction = my_model(embeddings)
            predictions.append(prediction)
            outputs.append(output)
    predictions = torch.cat(predictions, 0)
    outputs = torch.cat(outputs, 0)
    f1_micro = torchmetrics.functional.classification.multiclass_f1_score(
        predictions,
        outputs,
        classes,
        average='micro'
    )
    f1_macro = torchmetrics.functional.classification.multiclass_f1_score(
        predictions,
        outputs,
        classes,
        average='macro'
    )
    print(f"  Error: F1 micro: {f1_micro}, F1 macro: {f1_macro} \n")


def train_epoch(
    beats_model: BEATs.BEATs,
    my_model: classifier.NeuralNetwork1,
    loss_function: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    training_data: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    validation_data: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    classes: int,
    sample_rate: int
):
    my_model.train()
    step = 0
    for inputs, outputs in tqdm.tqdm(training_data):
        inputs, outputs = audio_processing.combine_samples(inputs, outputs)
        inputs = audio_processing.augment(inputs, sample_rate)
        
        padding_mask = torch.zeros(inputs.shape).bool()
        representation = beats_model.extract_features(inputs, padding_mask=padding_mask)[0]
        embeddings = representation.mean(dim=1)
        
        predictions = my_model(embeddings)
        loss = loss_function(predictions, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
    
    evaluate(
        beats_model,
        my_model,
        validation_data,
        classes,
        sample_rate
    )
    return step


def setup_beats(model_file: str):
    checkpoint = torch.load(model_file)
    config = BEATs.BEATsConfig(checkpoint['cfg'])
    model = BEATs.BEATs(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def main(parameters):
    if not gpu.test_gpu():
        return
    
    dataset_root = parameters.dataset_root
    checpoints_dir = parameters.checkpoints
    comet_key = parameters.comet_key
    beats_model_path = parameters.beats_model
    
    with open(comet_key, "r") as f:
        conf = f.read().splitlines()
    
    api_key = conf[0]
    project_name = conf[1]
    workspace = conf[2]
    
    hyperparameters = {
        'learning_rate': 1e-3,
        'batch_size': 128,
        'epochs': 100
    }
    
    print("Gathering dataset...")
    dataset_data = data.get_data(dataset_root)
    
    print("Setting embedding model...")
    beats_model = setup_beats(beats_model_path)
    
    print("Setting classifier...")
    my_model = classifier.NeuralNetwork1(768, 4)
    loss_function = torch.nn.CrossEntropyLoss()  # TODO: label_smoothing?
    optimizer = torch.optim.SGD(my_model.parameters(), lr=hyperparameters['learning_rate'])
    
    print("Training...")
    step = 0
    for epoch in range(int(hyperparameters['epochs'])):
        print(f"Epoch {epoch}")
        step += train_epoch(
            beats_model,
            my_model,
            loss_function,
            optimizer,
            dataset_data.training_set,
            dataset_data.validation_set,
            dataset_data.unique_labels,
            dataset_data.sample_rate
        )
        
        if epoch % 40 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': my_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                os.path.join(checpoints_dir, f'checkpoint_{epoch}.pt2')
            )
    
    print("Testing...")
    evaluate(
        beats_model,
        my_model,
        dataset_data.test_set,
        dataset_data.unique_labels,
        dataset_data.sample_rate
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument('--dataset_root', required=True)
    _ = parser.add_argument('--checkpoints', required=True)
    _ = parser.add_argument('--comet_key')
    _ = parser.add_argument('--beats_model', required=True)
    parameters = parser.parse_args()
    main(parameters)