import torch
import torch.nn as nn
import argparse
from batch_renorm2d import BatchRenormalization2D


def parse():
    args = argparse.ArgumentParser()
    args.add_argument("--num_feats", default=64, type=int)
    return args.parse_args()


if __name__ == "__main__":
    params = parse()
    bn = nn.BatchNorm2d(params.num_feats)
    brn = BatchRenormalization2D(params.num_feats, bn.state_dict())
    inputs = torch.randn(16, params.num_feats, 16, 16)
    # Initially, batch renorm is same as batch norm
    assert bn(inputs) == brn(inputs)
