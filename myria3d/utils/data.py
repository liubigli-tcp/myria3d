from typing import Optional, Union
import torch


def estimate_sample_weights(dataset,
                            class_weights: Optional[torch.tensor] = None,
                            reduction: Union[str, callable] ="mean",
                            normalize: bool = True):
    """
    :param dataset:
    :param class_weights:
    :param reduction:
    :param normalize:
    :return:
    """

    samples_scores = torch.zeros(len(dataset))

    for i, data in enumerate(dataset):
        if data is not None:
            y = data.y
            if reduction == "mean":
                samples_scores[i] = class_weights[y].mean()
            elif reduction == "max":
                samples_scores[i] = class_weights[y].max()
            elif reduction == "min":
                samples_scores[i] = class_weights[y].min()
            elif reduction == "sum":
                samples_scores[i] = class_weights[y].sum()
            else:
                samples_scores[i] = reduction(class_weights[y])

    if normalize:
        return samples_scores / samples_scores.sum()
    else:
        return samples_scores