from typing import Protocol

import torch


class ModelProtocol(Protocol):
    def __call__(self, image_batch: torch.Tensor) -> torch.Tensor:
        ...
