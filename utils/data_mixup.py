from typing import override
from audiomentations.core.transforms_interface import BaseWaveformTransform
import numpy


class BatchDataMixup(BaseWaveformTransform):
    supports_multichannel = True
    
    def __init__(
        self,
        alpha: float = 8,
        beta: float = 1,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.alpha: float = alpha
        self.beta: float = beta
        setattr(self.parameters, 'lambda', 1.0)

    @override
    def randomize_parameters(self, samples: numpy.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            setattr(self.parameters, 'lambda', numpy.random.beta(self.alpha, self.beta))

    @override
    def apply(self, samples: numpy.ndarray, batch: numpy.ndarray, length: numpy.ndarray):  # pyright: ignore[reportIncompatibleMethodOverride]
        ix = numpy.random.randint(0, batch.shape[0])
        minlen = min(samples.shape[-1], length[ix])
        lam: float = getattr(self.parameters, 'lambda')
        samples[:minlen] = lam * samples[:minlen] + (1 - lam) * batch[ix, :minlen]
        return samples