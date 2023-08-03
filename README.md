# THE ULTIMATE [BATCH RENORMALIZATION](https://arxiv.org/abs/1702.03275)

## INTRODUCTION

Batch normalization is everywhere in deep learning. It primarily consists of five parameters: 
scalable `gamma`,
shiftable `beta`,
running average,
running variance,
and counter on tracked batches.
Batch renormalization was firstly introduced in 2017 to address mini-batch training and non-iid sampling issues.

## Highlights

According to [PyTorch implementation](https://pytorch.org/docs/main/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d), the variance calculation is tricky. We always use biased estimator 
in forwarding and unbiased estimator to update running variance.
In addition to this, for non-iid sampling, we always need to manually re-sampling in forwarding to make the sampling iid.

## Core Ideas

Batch re-normalization is modified based on batch norm, with a slight loose on `d_max` and `r_max`,
while in batch normalization, these two are immutable.

## Implementation

Everything is on a premise of `affine=True` and `track_running_stats=True`.