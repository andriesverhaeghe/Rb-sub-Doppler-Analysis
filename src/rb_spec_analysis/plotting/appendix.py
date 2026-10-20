"""Plotting for the Appendix of Rb spectroscopy analysis.

Every function takes already-processed data (see analysis.py and
schemas/analysis_schemas.py) and draws it. Nothing is fitted, masked or
normalised here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import rcParams

PURPLE, YELLOW, BLUE, RED, OLIVE = "#68246D", "#FFD53A", "#00AEEF", "#BE1E2D", "#AFA961"

rcParams.update({"axes.labelsize": 20, "xtick.labelsize": 15, "ytick.labelsize": 15})
rcParams["axes.prop_cycle"] = cycler(color=[PURPLE, YELLOW, BLUE, RED, OLIVE])

FREQ_LABEL = "Frequency with Arbitrary Offset (GHz)"

def basic_1var_signal_time(signal_x: np.ndarray, signal_y: np.ndarray, *args):
    plt.plot(signal_x, signal_y, *args)
    plt.xlabel("Time (ms)")
    plt.ylabel("Signal (V)")

    plt.show()


def basic_2var_signal_time(
    signal_x1: np.ndarray,
    signal_y1: np.ndarray,
    signal_x2: np.ndarray,
    signal_y2: np.ndarray,
    *args,
    **kwargs,
):
    plt.plot(signal_x1, signal_y1, *args)
    plt.plot(signal_x2, signal_y2)

    plt.xlabel(kwargs.get("xlabel") or "Time (ms)")
    plt.ylabel(kwargs.get("ylabel") or "Signal (V)")

    plt.show()


def figure_9(
    signal_x: np.ndarray,
    signal_y: np.ndarray,
    coeffs: np.ndarray | None = None,
    func: Callable | None = None,
):

    plt.plot(signal_x, signal_y)

    if coeffs is not None and func is not None:
        plt.plot(signal_x, func(signal_x, *coeffs), linestyle="dashed", color="black", zorder=2)

    plt.xlabel(FREQ_LABEL)
    plt.ylabel("Photodiode \n Output (V)")
    plt.ylim(
        0,
    )
    plt.tick_params(axis="both", which="major", labelsize=22)

    plt.show()


def plot_x_points(signal_x: np.ndarray, signal_y: np.ndarray, x_ones: Iterable):
    for i in x_ones:
        plt.scatter(i[0], i[1], s=100, marker="x", zorder=2, color="red")
        plt.plot(signal_x, signal_y, zorder=1)

    plt.show()


def scatter_and_plot(
    scatter_x: np.ndarray,
    scatter_y: np.ndarray,
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    *args,
    **kwargs,
):

    plt.scatter(scatter_x, scatter_y, color=kwargs.get("color") or "#BE1E2D")
    plt.plot(plot_x, plot_y)
    plt.show()
