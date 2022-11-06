"""
File: table_1.py
Project: POLICE: PROVABLY OPTIMAL LINEAR CONSTRAINT ENFORCEMENT FOR DEEP NEURAL NETWORKS
Link: https://arxiv.org/abs/2211.01340
-----
# Copyright (c) Randall Balestriero
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch as ch
from tqdm import tqdm
import time
from utils import ConstrainedNetwork, UnconstrainedNetwork


def print_computation_time(D, width, depth, N=1024):

    training_steps = 1024  # number of training steps

    # we create the constraints which in this case correspond to the simplex region
    # of the corresponding ambiant space
    constraints = ch.cat([ch.zeros(1, D), ch.eye(D)]).cuda() / np.sqrt(D)
    constraints = constraints.detach().clone()

    # and we create some dummy target
    points = ch.randn(N, D).cuda() / np.sqrt(D)
    target = ch.cos(5 * ch.linalg.norm(points, axis=1)).cuda()
    target -= target.mean()
    target /= target.abs().max()

    all_times = np.zeros((2, training_steps))
    act = ch.nn.functional.leaky_relu
    # now we run the two baselines
    for constrained in [0, 1]:

        # model and optimizer definition
        if constrained:
            model = ConstrainedNetwork(constraints, D, depth, width, act)
        else:
            model = UnconstrainedNetwork(D, depth, width, act)
        model = model.cuda()
        output_layer = ch.nn.Linear(width, 1).cuda()

        params = list(model.parameters()) + list(output_layer.parameters())
        optim = ch.optim.AdamW(params, 0.0001)

        # training
        with tqdm(total=training_steps // 100) as pbar:
            for i in range(training_steps):
                t = time.time()
                output = output_layer(model(points))[:, 0]
                loss = ch.nn.functional.mse_loss(output, target)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                all_times[constrained, i] = (time.time() - t) * 1000
                optim.step()
                if i % 100 == 0:
                    pbar.update(1)
                    pbar.set_description(f"Loss {loss.item()}")

    mean_0 = np.round(np.mean(all_times[0]), 2)
    mean_1 = np.round(np.mean(all_times[1]), 2)
    std_0 = np.round(np.std(all_times[0]), 2)
    std_1 = np.round(np.std(all_times[1]), 2)

    print(f"Time (ms.) for input dim:{str(D)}, width:{width}, depth:{depth}")
    print(f"\t- unconstrained {mean_0} $\pm$ {std_0}")
    print(f"\t- POLICE: {mean_1} $\pm$ {std_1}")
    print(f"\t- slow-down factor: {mean_1/mean_0}")


if __name__ == "__main__":

    print_computation_time(28 * 28, width=128, depth=4)

    print_computation_time(2, width=512, depth=16)

    print_computation_time(3 * 32 * 32, width=4096, depth=2)
