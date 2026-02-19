import argparse
import dataclasses
import comet_ml
import comet_ml.integration
import comet_ml.integration.pytorch
import torch
import torchmetrics
import torchmetrics.functional.classification
import tqdm

import perch
import data
import classifier


@dataclasses.dataclass
class ModelLearning:
    model: classifier.NeuralNetwork1
    loss_function: torch.nn.CrossEntropyLoss
    optimizer: torch.optim.SGD


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
    learning: ModelLearning,
    perch: perch.PerchV2,
    data: torch.utils.data.DataLoader[data.ZabeDataset],
    experiment: comet_ml.CometExperiment
):
    learning.model.eval()
    predictions = []
    outputs = []
    
    with torch.no_grad():
        for input, output in tqdm.tqdm(data):
            embedding, _, _ , _ = perch(input)
            prediction = learning.model(embedding)
            predictions.append(prediction)
            outputs.append(output)
    predictions = torch.cat(predictions, 0)
    outputs = torch.cat(outputs, 0)
    f1_micro = torchmetrics.functional.classification.multiclass_f1_score(
        predictions,
        outputs,
        data.dataset.unique, # type: ignore
        average='micro'
    )
    f1_macro = torchmetrics.functional.classification.multiclass_f1_score(
        predictions,
        outputs,
        data.dataset.unique, # type: ignore
        average='macro'
    )
    
    experiment.log_metric("f1_micro", f1_micro)
    experiment.log_metric("f1_macro", f1_macro)
    
    print(f"  Error: F1 micro: {f1_micro}, F1 macro: {f1_macro} \n")


def training(
    learning: ModelLearning,
    perch: perch.PerchV2,
    train_data: torch.utils.data.DataLoader[data.ZabeDataset],
    val_data: torch.utils.data.DataLoader[data.ZabeDataset],
    epochs: int,
    experiment: comet_ml.CometExperiment,
):
    comet_ml.integration.pytorch.watch(learning.model)
    step = 0
    
    for t in range(epochs):
        print(f"Epoch: {t}")
        experiment.log_current_epoch(t)
        
        learning.model.train()
        for inputs, outputs in tqdm.tqdm(train_data):
            embedding, _, _, _ = perch(inputs)
            prediction = learning.model(embedding)
            loss: torch.Tensor = learning.loss_function(prediction, outputs)
            loss.backward()
            learning.optimizer.step()
            learning.optimizer.zero_grad()
            
            experiment.log_metric("loss", loss.item(), step=step)
            step += 1
        
        evaluate(learning, perch, val_data, experiment)


def main(data_dir: str):
    if not test_gpu():
        return
    
    perchv2 = perch.load_perch_from_onnx("perch_v2.onnx")
    perchv2.eval()
    
    experiment = comet_ml.start(
        api_key="zdwRvLYZJ7dxhgmCOKHZBRIgC",
        project_name="zabe",
        workspace="grimpy101"
    )
    hyperparameters = {
        'learning_rate': 1e-3,
        'batch_size': 64,
        'epochs': 100
    }
    experiment.log_parameters(hyperparameters)
    
    print("Preparing datasets...")
    
    train_data, test_data, val_data = data.load_datasets(
        data_dir,
        int(hyperparameters['batch_size'])
    )
    
    print("Init model...")
    
    model = classifier.NeuralNetwork1(1536, train_data.dataset.unique) # type: ignore
    
    print("Init loss function and optimizer...")

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters['learning_rate'])
    
    learning = ModelLearning(model, loss_function, optimizer)
    
    print("Start training...")
    with experiment.train():
        training(
            learning,
            perchv2,
            train_data,
            val_data,
            hyperparameters['epochs'], # type: ignore
            experiment
        )
    with experiment.test():
        evaluate(
            learning,
            perchv2,
            test_data,
            experiment
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Zabe")
    parser.add_argument("--data_dir")
    arguments = parser.parse_args()
    
    main(arguments.data_dir)