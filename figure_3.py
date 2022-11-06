"""
File: figure_3.py
Project: POLICE: PROVABLY OPTIMAL LINEAR CONSTRAINT ENFORCEMENT FOR DEEP NEURAL NETWORKS
Link: https://arxiv.org/abs/2211.01340
-----
# Copyright (c) Randall Balestriero
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch as ch
from tqdm import tqdm
from utils import ConstrainedNetwork


def plot_training_evolution(
    width,
    depth,
    training_steps: int = 2000,
    activation: callable = ch.nn.functional.leaky_relu,
):

    # set up the triangle constraint
    constraints = np.array(
        [
            [-np.sqrt(2), -np.sqrt(2)],
            [0, 2],
            [np.sqrt(2), -np.sqrt(2)],
        ]
    )

    # and create the target data (corresponding to the rays setting)
    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    grid = ch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().cuda()

    angle = ch.angle(grid[:, 0] + 1j * grid[:, 1])
    target = ch.cos(6 * angle)

    target -= target.mean()
    target /= target.abs().max()

    # model and optimizer definition
    model = ConstrainedNetwork(constraints, 2, depth, width, activation).cuda()
    output_layer = ch.nn.Linear(width, 1).cuda()
    params = list(model.parameters()) + list(output_layer.parameters())
    optim = ch.optim.AdamW(params, 0.001)
    scheduler = ch.optim.lr_scheduler.StepLR(
        optim, step_size=training_steps // 4, gamma=0.3
    )

    # training
    targets = []
    stops = [5, 20, 60, training_steps - 1]  # this tells us which snapshots to keep
    with tqdm(total=training_steps // 100) as pbar:

        for i in range(training_steps):
            output = output_layer(model(grid))[:, 0]
            loss = ch.nn.functional.mse_loss(output, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()
            if i % 100 == 0:
                pbar.update(1)
                pbar.set_description(f"Loss {loss.item()}")
            # for plotting
            if i in stops:
                with ch.no_grad():
                    pred = output_layer(model(grid)).clamp(-1, 1).cpu().numpy()
                    targets.append(pred.reshape((100, 100)))

    constraints = np.concatenate([constraints, constraints[[0]]], 0)
    fig, axs = plt.subplots(
        1,
        1 + len(targets),
        sharex="all",
        sharey="all",
        figsize=(len(targets) * 5 + 5, 5),
    )
    levels = np.linspace(-1.0, 1.0, 12).round(2)
    target = target.reshape((100, 100)).cpu().numpy()
    axs[0].contourf(
        xx,
        yy,
        target,
        cmap="plasma",
        levels=levels,
    )
    for i, target in enumerate(targets):
        i += 1
        axs[i].contourf(xx, yy, target, cmap="plasma", levels=levels)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].scatter(constraints[:, 0], constraints[:, 1], c="k")
        axs[i].plot(constraints[:, 0], constraints[:, 1], c="k")

    plt.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.035, 0.035)
    plt.savefig(f"./figures/training_evolution.png")
    plt.close()


if __name__ == "__main__":

    width = 256  # width of network
    depth = 3  # depth of network
    plot_training_evolution(width, depth)
