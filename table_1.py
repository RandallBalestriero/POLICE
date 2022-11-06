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


def print_computation_time(D, width, depth, N):

    S = 1000  # number of training steps
    losses = {}
    constraints = ch.cat([ch.zeros(1, D), ch.eye(D)]).cuda() / np.sqrt(D)
    points = ch.randn(N, D).cuda() / np.sqrt(D)
    target = ch.cos(5 * ch.linalg.norm(points, axis=1)).cuda()

    target -= target.mean()
    target /= target.abs().max()

    all_times = np.zeros((2, S))
    for constrained in [0, 1]:

        # model and optimizer definition
        if constrained:
            model = ConstrainedNetwork(
                constraints.detach().clone(),
                D,
                depth,
                width,
                ch.nn.functional.leaky_relu,
            ).cuda()
        else:
            model = UnconstrainedNetwork(
                D, depth, width, ch.nn.functional.leaky_relu
            ).cuda()
        output_layer = ch.nn.Linear(width, 1).cuda()

        optim = ch.optim.AdamW(
            list(model.parameters()) + list(output_layer.parameters()),
            0.0001,
            weight_decay=1e-5,
        )
        scheduler = ch.optim.lr_scheduler.StepLR(optim, step_size=S // 4, gamma=0.3)

        # training
        with tqdm(total=S // 100) as pbar:
            for i in range(S):
                t = time.time()
                output = output_layer(model(points))[:, 0]
                loss = ch.nn.functional.mse_loss(output, target)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                all_times[constrained, i] = time.time() - t
                optim.step()
                scheduler.step()
                if i % 100 == 0:
                    pbar.update(1)
                    pbar.set_description(f"Loss {loss.item()}")
    print(
        f"Time for {str(D)}_{width}_{depth}: {np.mean(all_times[0]*1000)} \pm {np.std(all_times[0]*1000)}"
    )
    print(
        f"Time for {str(D)}_{width}_{depth}: {np.mean(all_times[1]*1000)} \pm {np.std(all_times[1]*1000)}"
    )
    print(
        f"Time for {str(D)}_{width}_{depth}: {np.mean(all_times[1])/np.mean(all_times[0])}"
    )


if __name__ == "__main__":

    print_computation_time(32 * 32 * 3, 4096, 6, 1024)
    print_computation_time(2, 256, 2, 1024)
    print_computation_time(2, 256, 8, 1024)

    print_computation_time(2, 64, 4, 1024)
    print_computation_time(2, 4096, 4, 1024)

    print_computation_time(28 * 28, 1024, 2, 1024)
    print_computation_time(28 * 28, 1024, 8, 1024)
