import argparse
import os
import torch
import torchmetrics
import torchmetrics.functional.classification
import tqdm

import comet_ml
import comet_ml.integration
import comet_ml.integration.pytorch

from . import data, classifier


def test_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False
    
    device_count = torch.cuda.device_count()
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    
    print(f"Found {device_count} CUDA devices")
    print(f"Current device: {device_index} ({device_name})")
    return True


def evaluate(
    model: classifier.NeuralNetwork1,
    data: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    classes_count: int,
    experiment: comet_ml.CometExperiment
):
    _ = model.eval()
    predictions = []
    outputs = []
    
    with torch.no_grad():
        for embeddings, labels in tqdm.tqdm(data):
            prediction = model(embeddings)
            predictions.append(prediction)
            outputs.append(labels)
    predictions = torch.cat(predictions, 0)
    outputs = torch.cat(outputs, 0)
    
    f1_micro = torchmetrics.functional.classification.multiclass_f1_score(
        predictions,
        outputs,
        classes_count,
        average='micro'
    )
    f1_macro = torchmetrics.functional.classification.multiclass_f1_score(
        predictions,
        outputs,
        classes_count,
        average='macro'
    )
    print(f"  Error: F1 micro: {f1_micro}, F1 macro: {f1_macro} \n")
    experiment.log_metric("f1_micro", f1_micro)
    experiment.log_metric("f1_macro", f1_macro)
    
    return f1_micro, f1_macro


def train(
    model: classifier.NeuralNetwork1,
    loss_function: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    data: data.DatasetData,
    epochs: int,
    checkpoint_dir: str,
    experiment: comet_ml.CometExperiment
):
    comet_ml.integration.pytorch.watch(model)
    
    step = 0
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        experiment.log_current_epoch(epoch)
        
        model.train()
        for embeddings, labels in tqdm.tqdm(data.training_set):
            prediction = model(embeddings)
            loss = loss_function(prediction, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            experiment.log_metric("loss", loss.item(), step=step)
            step += 1
        
        f1_micro, f1_macro = evaluate(
            model,
            data.validation_set,
            data.unique_labels,
            experiment
        )
        
        if epoch % 40 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1_micro': f1_micro,
                    'f1_macro': f1_macro
                },
                os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt2')
            )


def main(parameters):
    if not test_gpu():
        return
    
    dataset_root = parameters.dataset_root
    checpoints_dir = parameters.checkpoints
    
    with open("comet_api_key", "r") as f:
        conf = f.read().splitlines()
        
    api_key = conf[0]
    project_name = conf[1]
    workspace = conf[2]
    
    experiment = comet_ml.start(
        api_key=api_key,
        project_name=project_name,
        workspace=workspace
    )
    
    hyperparameters = {
        'learning_rate': 1e-3,
        'batch_size': 128,
        'epochs': 1
    }
    experiment.log_parameters(hyperparameters)
    
    print("Preparing datasets...")
    dataset_data = data.get_data(dataset_root)
    
    print("Init model...")
    model = classifier.NeuralNetwork1(1536, dataset_data.unique_labels)
    
    print("Init loss function and optimizer...")
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters['learning_rate'])
    
    print("Start training...")
    with experiment.train():
        train(
            model,
            loss_function,
            optimizer,
            dataset_data,
            hyperparameters['epochs'],  # pyright: ignore[reportArgumentType]
            checpoints_dir,
            experiment
        )
    
    with experiment.test():
        f1_micro, f1_macro = evaluate(
            model,
            dataset_data.test_set,
            dataset_data.unique_labels,
            experiment
        )
    
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument('--dataset_root')
    _ = parser.add_argument('--checkpoints')
    parameters = parser.parse_args()
    main(parameters)