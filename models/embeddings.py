import torch

from . import perch_v2
from .beats import BEATs


class EmbeddingsModel:
    def __init__(self, model: str, model_path: str, device: str = 'cuda') -> None:
        if model == 'perchv2':
            self.model = perch_v2.PerchV2Torch(model_path, device=device)
        elif model == 'beats':
            checkpoint = torch.load(model_path)
            config = BEATs.BEATsConfig(checkpoint['cfg'])
            m = BEATs.BEATs(config)
            m.load_state_dict(checkpoint['model'])
            m.eval()
            self.model = m
        else:
            raise Exception("Invalid model chosen!")
    
    @property
    def embedding_dims(self) -> int:
        if isinstance(self.model, perch_v2.PerchV2Torch):
            return 1536
        elif isinstance(self.model, BEATs.BEATs):
            return 768
        else:
            raise Exception("Invalid model chosen!")
    
    def get_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(self.model, BEATs.BEATs):
                padding_mask = torch.zeros(inputs.shape).bool()  # type: ignore
                representation = self.model.extract_features(inputs, padding_mask=padding_mask)[0]
                embeddings = representation.mean(dim=1)
            elif isinstance(self.model, perch_v2.PerchV2Torch):
                embeddings = self.model(inputs)["embedding"]
            else:
                raise Exception("Invalid model!")
        return embeddings