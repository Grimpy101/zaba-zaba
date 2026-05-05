import argparse
import tomllib
import os

import comet_ml
import comet_ml.integration
import comet_ml.integration.pytorch

import torch
import torchmetrics
import torchmetrics.functional.classification
import tqdm

from .utils import gpu
from .data import data_loading
from .models import embeddings, frog_classifier


def test(
    embedding_model: embeddings.EmbeddingsModel,
    classifier: frog_classifier.FrogClassifier,
    data: data_loading.Data,
    device: str,
    experiment: comet_ml.CometExperiment
):
    predictions = []
    outputs = []
    
    with torch.no_grad():
        for inputs, output in tqdm.tqdm(data.validation_loader):
            inputs, output = data.process_samples(inputs, output)
        
            inputs = inputs.to(device)
            output = output.to(device)
            
            embeddings = embedding_model.get_embeddings(inputs)
            prediction = classifier(embeddings)
            predictions.append(prediction)
            outputs.append(output)
    
    predictions = torch.cat(predictions, 0)  # type: ignore
    prediction_probabilities = torch.sigmoid(predictions)  # type: ignore
    outputs = torch.cat(outputs, 0)  # type: ignore
    outputs_binary = (outputs > 0.5).long()
    f1_micro = torchmetrics.functional.classification.multilabel_f1_score(
        prediction_probabilities,
        outputs_binary,
        num_labels=data.dataset.classes,
        average='micro'
    )
    f1_macro = torchmetrics.functional.classification.multilabel_f1_score(
        prediction_probabilities,
        outputs_binary,
        num_labels=data.dataset.classes,
        average='macro'
    )
    print(f"  Error: F1 micro: {f1_micro}, F1 macro: {f1_macro} \n")
    
    experiment.log_metric("f1_micro_test", f1_micro)
    experiment.log_metric("f1_macro_test", f1_macro)


def validate(
    embedding_model: embeddings.EmbeddingsModel,
    classifier: frog_classifier.FrogClassifier,
    data: data_loading.Data,
    device: str,
    experiment: comet_ml.CometExperiment
):
    predictions = []
    outputs = []
    
    with torch.no_grad():
        for inputs, output in tqdm.tqdm(data.validation_loader):
            inputs, output = data.process_samples(inputs, output)
        
            inputs = inputs.to(device)
            output = output.to(device)
            
            embeddings = embedding_model.get_embeddings(inputs)
            prediction = classifier(embeddings)
            predictions.append(prediction)
            outputs.append(output)
    
    predictions = torch.cat(predictions, 0)  # type: ignore
    prediction_probabilities = torch.sigmoid(predictions)  # type: ignore
    outputs = torch.cat(outputs, 0)  # type: ignore
    outputs_binary = (outputs > 0.5).long()
    f1_micro = torchmetrics.functional.classification.multilabel_f1_score(
        prediction_probabilities,
        outputs_binary,
        num_labels=data.dataset.classes,
        average='micro'
    )
    f1_macro = torchmetrics.functional.classification.multilabel_f1_score(
        prediction_probabilities,
        outputs_binary,
        num_labels=data.dataset.classes,
        average='macro'
    )
    print(f"  Error: F1 micro: {f1_micro}, F1 macro: {f1_macro} \n")
    
    experiment.log_metric("f1_micro_val", f1_micro)
    experiment.log_metric("f1_macro_val", f1_macro)


def training_epoch(
    embedding_model: embeddings.EmbeddingsModel,
    classifier: frog_classifier.FrogClassifier,
    loss_function: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    data: data_loading.Data,
    step: int,
    device: str,
    experiment: comet_ml.CometExperiment
):
    classifier.train()
    step = step
    for inputs, outputs in tqdm.tqdm(data.train_loader):
        inputs, outputs = data.process_samples(inputs, outputs)
        
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        
        embeddings = embedding_model.get_embeddings(inputs)
        predictions = classifier(embeddings)
        
        loss = loss_function(predictions, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        experiment.log_metric("loss", loss.item(), step=step)
        
        step += 1
    
    validate(embedding_model, classifier, data, device, experiment)
    return step


def main(parameters: dict):
    if not gpu.test_gpu():
        return
    device = 'cuda'
    
    with open(parameters['config'], 'rb', encoding='utf-8') as f:
        config: dict = tomllib.load(f)
    
    general_settings: dict = config['general']
    comet_settings: dict = config['comet']
    data_settings: dict = config['data']
    checkpoint_settings: dict = config['checkpoint']
    hyperparameters: dict = config['hyperparameters']
    
    experiment = comet_ml.start(
        api_key=comet_settings['key'],
        project_name=comet_settings['project'],
        workspace=comet_settings['workspace']
    )
    experiment.log_parameters(hyperparameters)
    
    print("Gathering dataset...")
    data = data_loading.Data(data_settings, hyperparameters)
    data.print_info()
    
    print("Setting embedding model...")
    model_type = general_settings['model']
    model_path = general_settings['model_path']
    embedding_model = embeddings.EmbeddingsModel(model_type, model_path)
    
    print("Setting classifier...")
    classifier = frog_classifier.FrogClassifier(
        embedding_model.embedding_dims,
        data.dataset.classes
    )
    classifier.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=hyperparameters['learning_rate'])
    
    print("Training...")
    with experiment.train():  # type: ignore
        comet_ml.integration.pytorch.watch(classifier)
        step = 0
        
        for epoch in range(hyperparameters['epochs']):
            print(f"Epoch {epoch}")
            experiment.log_current_epoch(epoch)
            
            step = training_epoch(
                embedding_model, classifier,
                loss_function, optimizer,
                data, step,
                device, experiment
            )
            
            if epoch % checkpoint_settings['checkpoint_epoch'] == 0:
                checkpoint_filepath = os.path.join(checkpoint_settings['checkpoint_dir'], f'checkpoint_{epoch}.pt2')
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                    checkpoint_filepath
                )
                print(f"Saved checkpoint for epoch {epoch} to file {checkpoint_filepath}")
    
    checkpoint_filepath = os.path.join(checkpoint_settings['checkpoint_dir'], 'checkpoint_final.pt2')
    torch.save(
        {
            'epoch': hyperparameters['epochs'],
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        checkpoint_filepath
    )
    print(f"Saved final checkpoint to file {checkpoint_filepath}")
    
    print("Testing...")
    with experiment.test():  # type: ignore
        test(
            embedding_model, classifier,
            data, device, experiment
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _ = parser.add_argument('--config', required=True)
    parameters = parser.parse_args()
    
    