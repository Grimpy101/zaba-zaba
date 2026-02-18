import torch


class NeuralNetwork1(torch.nn.Module):
    def __init__(self, in_dims: int, out_dims: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.flatten = torch.nn.Flatten()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(in_dims, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, out_dims),
        )

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        logits: torch.Tensor = self.stack(x)
        return logits
