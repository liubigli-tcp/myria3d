import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn.glob.glob import global_max_pool

from semantic_val.datamodules.datasets.lidar_transforms import get_subsampling_mask


def MLP(channels, batch_norm: bool = True):
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            if batch_norm
            else Seq(Lin(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ]
    )


class PointNet(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        hparams_net = hparams["net"]
        self.mlp1 = MLP(
            hparams_net["MLP1_channels"], batch_norm=hparams_net["batch_norm"]
        )
        self.mlp2 = MLP(
            hparams_net["MLP2_channels"], batch_norm=hparams_net["batch_norm"]
        )
        self.mlp3 = MLP(
            hparams_net["MLP3_channels"], batch_norm=hparams_net["batch_norm"]
        )
        self.lin = Lin(hparams_net["MLP3_channels"][-1], hparams_net["num_classes"])
        pi_init = hparams_net["pi_init"]
        a = 0
        b = -np.log((1 - pi_init) / pi_init)
        self.lin.bias = torch.nn.Parameter(
            torch.Tensor(
                [
                    a,
                    b,
                ]
            )
        )
        nn.init.xavier_normal_(self.lin.weight)

    def forward(self, batch):
        """
        Object batch is a PyG data.Batch, as defined in custom collate_fn.
        Tensors pos and x (features) are in long format (B*N, M) expected by pyG methods.
        """
        features = torch.cat([batch.pos, batch.x], axis=1)
        input_size = features.shape[0]
        subsampling_size = (batch.batch_x == 0).sum()

        # Pass through network layers
        f1 = self.mlp1(features)
        f2 = self.mlp2(f1)
        context_vector = global_max_pool(f2, batch.batch_x)
        expanded_context_vector = (
            context_vector.unsqueeze(1)
            .expand((-1, subsampling_size, -1))
            .reshape(input_size, -1)
        )
        Gf1 = torch.cat((expanded_context_vector, f1), 1)
        f3 = self.mlp3(Gf1)
        logits = self.lin(f3)

        return logits

    # TODO: define weights
    # def init_all_weights(self):
    #     for p in self.parameters():
    #         self.init_weights(p)
    #     self.lin.bias = torch.nn.Parameter(
    #         torch.Tensor(
    #             [
    #                 1 - self.percentage_buildings_train_val,
    #                 self.percentage_buildings_train_val,
    #             ]
    #         )
    #     )
    #     nn.init.xavier_normal_(self.lin.weight)

    # def init_weights(self, p):
    #     """Initialize weights of the Parameter p"""
    #     classname = p.__class__.__name__
    #     if classname.find("Linear") == 0 and hasattr(p, "weight"):
    #         gain = nn.init.calculate_gain("relu")
    #         nn.init.xavier_normal_(p, gain=gain)
