from typing import List, Union
import torch
from torch.utils.data import WeightedRandomSampler
from myria3d.utils.data import estimate_sample_weights


class ClassWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, dataset, class_weights: Union[torch.Tensor, List], replacement: bool = True):
        class_weights = torch.Tensor(class_weights)
        print(f"Class weight: {class_weights}")
        sample_weights = estimate_sample_weights(dataset, class_weights=class_weights)
        super().__init__(weights=sample_weights, num_samples=len(dataset), replacement=replacement)