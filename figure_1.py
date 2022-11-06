"""
File: figure_1.py
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


def plot_classification_case(
    width: int, depth: int, constraints: np.ndarray, training_steps=2000
) -> None:

    N = 1024

    constraints = ch.from_numpy(constraints).float().cuda()

    # data generation
    points = []
    for theta in np.linspace(0, 2 * np.pi, N):
        r = 1 + 0.2 * np.cos(theta * 10)
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        points.append([x, y])
        points.append([x * 1.3, y * 1.3])

    points = ch.from_numpy(np.stack(points)).float().cuda()
    points += ch.randn_like(points) * 0.025
    target = ch.from_numpy(np.tile(np.arange(2), N)).long().cuda()

    # model and optimizer definition
    model = ConstrainedNetwork(
        constraints, 2, depth, width, ch.nn.functional.leaky_relu
    ).cuda()
    output_layer = ch.nn.Linear(width, 1).cuda()

    optim = ch.optim.AdamW(
        list(model.parameters()) + list(output_layer.parameters()),
        0.001,
        weight_decay=1e-5,
    )
    scheduler = ch.optim.lr_scheduler.StepLR(
        optim, step_size=training_steps // 4, gamma=0.3
    )

    # training
    with tqdm(total=training_steps // 100) as pbar:
        for i in range(training_steps):
            output = output_layer(model(points))[:, 0]
            loss = ch.nn.functional.binary_cross_entropy_with_logits(
                output, target.float()
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()
            if i % 100 == 0:
                pbar.update(1)
                pbar.set_description(f"Loss {loss.item()}")

    # plotting
    domain_bound = 1.8
    with ch.no_grad():
        xx, yy = np.meshgrid(
            np.linspace(-domain_bound, domain_bound, 150),
            np.linspace(-domain_bound, domain_bound, 150),
        )
        grid = ch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().cuda()
        pred = output_layer(model(grid)).cpu().numpy()
        grid = grid.cpu().numpy()
        points = points.cpu().numpy()
        target = target.cpu().numpy()
        constraints = constraints.cpu().numpy()

    plt.figure(figsize=(5, 5))

    # plot training data
    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=["purple" if l == 0 else "orange" for l in target],
        alpha=0.45,
    )

    # plot our decision boundary
    plt.contour(
        xx,
        yy,
        pred[:, 0].reshape((150, 150)),
        levels=[0],
        colors=["red"],
        linewidths=[4],
    )

    # plot the user-defined constraints
    plt.scatter(constraints[:, 0], constraints[:, 1], c="k")
    constraints = np.concatenate([constraints, constraints[[0]]], 0)
    plt.plot(constraints[:, 0], constraints[:, 1], c="k", linewidth=2)

    # small beautifying process and figure saving
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.035, 0.035)
    plt.savefig(f"figures/constrained_classification.png")
    plt.close()


if __name__ == "__main__":
    # we need to specify the constraint in term of its vertices, here
    # four vertices in dimension 2
    constraints = np.array([[0.0, 0.0], [1.5, 0.0], [1.5, 1.5], [0.0, 1.5]])

    # and let's define a simple MLP to solve the task
    width = 128
    depth = 3

    plot_classification_case(width, depth, constraints)
