"""
File: utils.py
Project: POLICE: PROVABLY OPTIMAL LINEAR CONSTRAINT ENFORCEMENT FOR DEEP NEURAL NETWORKS
Link: https://arxiv.org/abs/2211.01340
-----
# Copyright (c) Randall Balestriero
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import torch as ch


@ch.jit.script
def enforce_constraint_forward(
    x: ch.Tensor, W: ch.Tensor, b: ch.Tensor, C: int
) -> ch.Tensor:
    """Perform a forward pass on the given `x` argument which contains both the `C` vertices
    describing the region `R` onto which the DNN is constrained to stay affine, and the mini-batch

    Args:
        x (ch.Tensor): vertices and inputs to be forward, the first `C` rows contain the indices
        W (ch.Tensor): weights used for the linear mapping of the layer
        b (ch.Tensor): biases used for the linear mapping of the layer
        C (int): number of vertices describing the region

    Returns:
        ch.Tensor: the forwarded vertices and inputs
    """
    # pre-activation for everyone (data + constraints)
    # shape is thus (N + C, K) with K the output dim
    # with W RD:-> RK
    # we do not yet add the bias
    h = x @ W.T + b
    V = h[-C:]
    # now we check which constraints are not all agreeing
    # agreement will be of shape (K,) and agreement[k] tells
    # us what is the majority sign for output dim k
    with ch.no_grad():
        # this is true if positive is majority sign
        agreement = V > 0

        # select which units actually need intervention
        invalid_ones = agreement.all(0).logical_not_().logical_and_(agreement.any(0))

        # compute the majority sign
        sign = agreement[:, invalid_ones].half().sum(0).sub_(C / 2 + 1e-6).sign_()

    # look by how much do we have to shift each hyper-plane so that
    # all constraints have the majority sign
    extra_bias = (V[:, invalid_ones] * sign).amin(0).clamp(max=0) * sign
    h[:, invalid_ones] -= extra_bias
    return h


class ConstrainedLayer(ch.nn.Linear):
    def forward(self, x, C):
        return enforce_constraint_forward(x, self.weight, self.bias, C)


class ConstrainedNetwork(ch.nn.Module):
    def __init__(
        self, constraints, in_features, depth, width, nonlinearity, last_width=None
    ):
        super().__init__()
        self.register_buffer("depth", ch.as_tensor(depth))
        self.register_buffer("constraints", ch.as_tensor(constraints).float())
        self.nonlinearity = nonlinearity
        self.layer0 = ConstrainedLayer(in_features, width)
        if last_width is None:
            last_width = width
        for i in range(1, depth):
            setattr(
                self,
                f"layer{i}",
                ConstrainedLayer(width, last_width if i == (depth - 1) else width),
            )

    def forward(self, x):
        with ch.no_grad():
            x = ch.cat([x, self.constraints.detach()], 0)
        C = self.constraints.size(0)
        for i in range(self.depth):
            x = getattr(self, f"layer{i}")(x, C)
            x = self.nonlinearity(x)
        return x[:-C]


class UnconstrainedNetwork(ch.nn.Module):
    def __init__(self, in_features, depth, width, nonlinearity):
        super().__init__()
        self.register_buffer("depth", ch.as_tensor(depth))
        self.nonlinearity = nonlinearity
        self.layer0 = ch.nn.Linear(in_features, width)
        for i in range(1, depth):
            setattr(
                self,
                f"layer{i}",
                ch.nn.Linear(width, width),
            )

    def forward(self, x):
        for i in range(self.depth):
            x = getattr(self, f"layer{i}")(x)
            x = self.nonlinearity(x)
        return x
