"""
File: figure_2.py
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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from utils import ConstrainedNetwork


def plot_multicase(
    width: int,
    depth: int,
    function: str,
    constraint: str,
    training_steps: int = 2000,
    activation: callable = ch.nn.functional.leaky_relu,
):
    """train and plot the POLICE constrained DNN and the unconstrained cased for a few
    different target functions and type of regions to use for the constraint

    Args:
        width (int): width of the MLP to use for fitting
        depth (int): depth of the MLP to use for fitting
        function (str): name of the target function, for now only supports `wave`, `ways` and `spirale`
        constraint (str): name of the region to use for constraints, for now only supports `triangle`, `polygon` and `circle
        training_steps (int, optional): Number of steps for training of the DNN. Defaults to 10000.
        activation (callable, optional): activation function for the MLP. Defaults to ch.nn.functional.leaky_relu.
    """

    domain = 4  # defines the [-bounds,bound]^2 domain of the DNN

    # we first define the vertices of the region that needs to be
    # constrained
    if constraint == "polygon":
        inset_domain = [1.5, 1]
        constraints = np.array(
            [
                [-1.5, 1],
                [-0.5, 2],
                [0.5, 2],
                [1.5, 1],
                [1.5, -0.5],
                [0.5, -1.5],
                [-0.5, -1.5],
                [-1.5, -0.5],
            ]
        )
    elif constraint == "triangle":
        constraints = np.array(
            [
                [-np.sqrt(2), -np.sqrt(2)],
                [0, 2],
                [np.sqrt(2), -np.sqrt(2)],
            ]
        )
        inset_domain = [np.sqrt(2), -np.sqrt(2)]
    else:
        inset_domain = [np.sqrt(2), -np.sqrt(2)]
        constraints = []
        for x in np.linspace(-1, 1, 25):
            x = np.sin(np.pi * x / 2) * 2
            constraints.append([x, np.sqrt(4 - x**2)])
        for x in np.linspace(-1, 1, 25)[::-1]:
            x = np.sin(np.pi * x / 2) * 2
            constraints.append([x, -np.sqrt(4 - x**2)])
        constraints = np.stack(constraints)

    # input space grid and target function definition
    xx, yy = np.meshgrid(
        np.linspace(-domain, domain, 100),
        np.linspace(-domain, domain, 100),
    )
    grid = ch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().cuda()
    if function == "wave":
        target = ch.cos(grid[:, 0] * 3) * ch.cos(grid[:, 1])
    elif function == "rays":
        angle = ch.angle(grid[:, 0] + 1j * grid[:, 1])
        target = ch.cos(6 * angle)
    else:
        angle = ch.angle(grid[:, 0] + 1j * grid[:, 1])
        radius = (grid[:, 0] ** 2 + grid[:, 1] ** 2).sqrt()
        target = ch.cos(6 * radius + angle)
    target -= target.mean()
    target /= target.abs().max()

    # model and optimizer definition
    model = ConstrainedNetwork(constraints, 2, depth, width, activation).cuda()
    output_layer = ch.nn.Linear(width, 1).cuda()
    params = list(model.parameters()) + list(output_layer.parameters())
    optim = ch.optim.AdamW(params, 0.0005)
    scheduler = ch.optim.lr_scheduler.StepLR(
        optim, step_size=training_steps // 3, gamma=0.3
    )

    # training
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
                pbar.set_description(f"{function}-{constraint}, Loss {loss.item()}")

    # plotting
    with ch.no_grad():
        output = output_layer(model(grid)).clamp(-1, 1).cpu().numpy()
        output = output.reshape((100, 100))
        target = target.reshape((100, 100)).cpu().numpy()

    fig, axs = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(16, 5))
    levels = np.linspace(-1.0, 1.0, 12).round(2)
    im = axs[0].contourf(xx, yy, target, cmap="plasma", levels=levels)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].contourf(xx, yy, output, cmap="plasma", levels=levels)
    axs[1].scatter(constraints[:, 0], constraints[:, 1], c="k")
    constraints = np.concatenate([constraints, constraints[[0]]], 0)
    axs[1].plot(constraints[:, 0], constraints[:, 1], c="k")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    # inset axes....
    axins = axs[1].inset_axes([1.05, 0.1, 0.85, 0.85])
    axins.contourf(xx, yy, output, cmap="plasma", levels=levels)
    axins.scatter(constraints[:, 0], constraints[:, 1], c="k")
    axins.plot(constraints[:, 0], constraints[:, 1], c="k")
    # sub region of the original image
    axins.set_xlim(inset_domain[0] - 1, inset_domain[0] + 1)
    axins.set_ylim(inset_domain[1] - 1, inset_domain[1] + 1)
    axins.set_xticks([])
    axins.set_yticks([])

    for d in ["left", "right", "top", "bottom"]:
        axins.spines[d].set_linewidth(3)
        axins.spines[d].set_color("tab:blue")

    box, c1, c2 = mark_inset(
        axs[1],
        axins,
        loc1=2,
        loc2=4,
        lw=0.3,
        fc="none",
        ec="tab:blue",
        zorder=200,
    )
    box.set_linewidth(3)
    for c in [c1, c2]:
        c.set_linestyle(":")
        c.set_linewidth(3)
        c.set_color("tab:blue")

    # add a colorbar, well positioned and rescaled
    cbar_ax = fig.add_axes([0.005, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    for y in cbar.ax.get_yticklabels():
        y.set_fontweight(600)

    plt.subplots_adjust(0.08, 0.01, 0.7, 0.99, 0.035, 0.035)
    plt.savefig(f"./figures/regression_{function}_{constraint}.png")
    plt.close()


if __name__ == "__main__":

    width = 256  # width of network
    depth = 4  # depth of network

    for function in ["wave", "rays", "spiral"]:
        for constraint in ["polygon", "triangle", "circle"]:
            plot_multicase(width, depth, function, constraint)
