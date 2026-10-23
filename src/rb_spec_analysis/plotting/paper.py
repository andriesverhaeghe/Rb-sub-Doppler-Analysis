"""Plotting for the Rb spectroscopy analysis.

Every function takes already-processed data (see analysis.py and
schemas/analysis_schemas.py) and draws it. Nothing is fitted, masked or
normalised here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

from rb_spec_analysis.schemas.analysis_schemas import (
    Linearised,
    NormalisedDoppler,
    OffResonance,
    SmoothedSubDoppler,
    SubDopplerPeaks,
    TheoryFit,
)
from rb_spec_analysis.utils.data import CroppedData
from rb_spec_analysis.utils.params import Params
from rb_spec_analysis.utils.polynomials import poly1, poly5, poly7

PURPLE, YELLOW, BLUE, RED, OLIVE = "#68246D", "#FFD53A", "#00AEEF", "#BE1E2D", "#AFA961"

rcParams.update({"axes.labelsize": 20, "xtick.labelsize": 15, "ytick.labelsize": 15})
rcParams["axes.prop_cycle"] = cycler(color=[PURPLE, YELLOW, BLUE, RED, OLIVE])

FREQ_LABEL = "Frequency with Arbitrary Offset (GHz)"
params = Params()

# --------------------------------------------------------------------------- #
# Raw signals and cropping
# --------------------------------------------------------------------------- #
def plot_raw_signals(
    time, normalised_etalon, sub_doppler, *, doppler: np.ndarray | None = None
) -> None:
    plt.plot(time, normalised_etalon)

    if doppler is not None:
        plt.plot(time, doppler)

    plt.plot(time, sub_doppler)
    plt.xlabel("Time (ms)")
    plt.title("Select two points either side of region")

    print(r'Select region of interest')
    crop_points = np.array(plt.ginput(2))

    x_cropped = crop_points[:,0]

    plt.show(block=False)
    plt.close()

    params.save_param("X_CROP", x_cropped)


def plot_crop_selection(time, normalised_etalon, normalised_doppler, sub_doppler, x_crop) -> None:
    for x in x_crop:
        plt.vlines(x, 0, 2, linestyle="dashed", color="black")

    plt.plot(time, normalised_etalon, zorder=1, color=BLUE)
    plt.plot(time, normalised_doppler, zorder=1, color=PURPLE)
    plt.plot(time, sub_doppler / sub_doppler.max() + 1, zorder=1, color=RED)  # stacked for display

    plt.xlabel("Time (ms)")
    plt.ylabel("Photodiode Output (V)")
    plt.show()


def plot_cropped_signals(cropped: CroppedData) -> None:
    t_ms = cropped.Cropped_Time * 1e3

    _fig, ax = plt.subplots(3, sharex=True, figsize=(8, 7))
    ax[0].plot(t_ms, cropped.Cropped_Etalon, color=BLUE)
    ax[1].plot(t_ms, cropped.Cropped_Doppler, color=PURPLE)
    ax[2].plot(t_ms, cropped.Cropped_Sub_Doppler, color=RED)

    ax[2].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Photodiode Output (V)", fontsize=20, labelpad=10)
    ax[0].set_yticks([0, 0.5, 1])
    ax[2].set_yticks([0.4, 0.6])
    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=20)
    plt.show()


# --------------------------------------------------------------------------- #
# Etalon linearisation
# --------------------------------------------------------------------------- #
def plot_etalon_peaks(cropped: CroppedData, lin: Linearised) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(cropped.Cropped_Time * 1e3, cropped.Cropped_Etalon, zorder=1, color=BLUE)
    plt.plot(lin.peak_time_ms, lin.etalon_peaks, "x", markersize=10, zorder=2, color=RED)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.show()


def plot_peak_positions(lin: Linearised) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(lin.peak_number, lin.peak_time_ms, "o")
    plt.plot(lin.peak_number, poly1(lin.peak_number, *lin.peak_coeffs))
    plt.xticks(np.arange(lin.peak_number.min(), lin.peak_number.max() + 1, 2.0))
    plt.xlabel("Etalon Peak Number")
    plt.ylabel("Etalon Peak Position (ms)")
    plt.show()


def plot_peak_difference(cropped: CroppedData, lin: Linearised) -> None:
    t_ms = cropped.Cropped_Time * 1e3

    plt.figure(figsize=(6, 4))
    plt.plot(lin.peak_time_ms, lin.difference, "o")
    plt.plot(t_ms, poly7(t_ms, *lin.difference_coeffs))
    plt.xlabel("Peak Position (ms)")
    plt.ylabel("Difference (ms)")
    plt.show()


def plot_linearised_etalon(cropped: CroppedData, lin: Linearised) -> None:
    plt.plot(lin.zeroed_linear_time, cropped.Cropped_Etalon)
    plt.xlabel("Linearised Time (ms)")
    plt.ylabel("Normalised signal (V)")
    plt.show()

    freq_ghz = lin.product / 1e3
    plt.figure(figsize=(6, 4))
    plt.plot(freq_ghz, cropped.Cropped_Etalon)
    if cropped.Cropped_Doppler is not None:  # Doppler data is optional
        plt.plot(freq_ghz, cropped.Cropped_Doppler)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Signal (V)")
    plt.show()


def figure_8(cropped: CroppedData, lin: Linearised) -> None:
    t_ms = cropped.Cropped_Time * 1e3
    freq_ghz = lin.product / 1e3
    n = len(lin.peak_number)

    fig = plt.figure(figsize=(9 * 1.5, 7 * 1.3), constrained_layout=True)
    gs = GridSpec(2, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, :4])
    ax2 = fig.add_subplot(gs[0, 4:6])
    ax3 = fig.add_subplot(gs[1, 4:6])
    ax4 = fig.add_subplot(gs[1, :4])

    ax1.plot(t_ms, cropped.Cropped_Etalon, color=BLUE)
    ax1.plot(lin.peak_time_ms, lin.etalon_peaks, "x", linewidth=5, zorder=2, color="black")
    ax2.scatter(
        lin.peak_number + 1, lin.peak_time_ms, s=4, color="black", marker="x", zorder=2, lw=15
    )
    ax2.plot(
        lin.peak_number + 1,
        poly1(lin.peak_number, *lin.peak_coeffs),
        color="black",
        linestyle="dashed",
        zorder=1,
    )
    ax3.scatter(lin.peak_time_ms, lin.difference, s=4, marker="x", color="black", zorder=2, lw=15)
    ax3.plot(t_ms, poly7(t_ms, *lin.difference_coeffs), color="black", zorder=1, linestyle="dashed")
    ax4.plot(freq_ghz, cropped.Cropped_Etalon, color=BLUE)

    ax1.set_ylim(0, 1.15)
    ax4.set_ylim(0, 1.15)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Normalised\n Etalon Signal", fontsize=20)
    ax2.set_ylabel("Time (ms)", fontsize=20)
    ax2.set_xlabel("Peak Number", fontsize=20)
    ax2.set_xticks([2, 4, 6, 8, 10, 12, 14])
    ax2.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
    ax3.set_xticks([0, 20, 40, 60, 80, 100, 120, 140])
    ax3.set_xlabel("Time (ms)", fontsize=20)
    ax3.set_ylabel("Difference (ms)", fontsize=20)
    ax4.set_xlabel(FREQ_LABEL)
    ax4.set_ylabel("Normalised \n Etalon Signal")

    for i in range(n):
        ax1.annotate(str(i + 1), (lin.peak_time_ms[i], lin.etalon_peaks[i] + 0.05), fontsize=16)

    arrow = {"arrowstyle": "<->"}
    for lo, hi in [(0.10, 0.84), (4.58, 5.30), (9.04, 9.78)]:
        ax4.annotate(text="", xy=(lo, 0.4), xytext=(hi, 0.4), arrowprops=arrow)
    for i, j in [(0, 1), (6, 7), (n - 2, n - 1)]:
        ax1.annotate(
            text="",
            xy=(lin.peak_time_ms[i], 0.4),
            xytext=(lin.peak_time_ms[j], 0.4),
            arrowprops=arrow,
        )

    for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(axis="both", which="major", labelsize=18)
    plt.show()


# --------------------------------------------------------------------------- #
# Sub-Doppler background and absolute frequency
# --------------------------------------------------------------------------- #
def sub_doppler_selection(detuning: np.ndarray, sub_dopp: np.ndarray):
    plt.plot(detuning, sub_dopp)

    plt.title("Select six points for linear region")
    x_sub_dopp = np.array(plt.ginput(6))

    plt.show(block=False)
    plt.close()

    params.save_param("X_SUB_DOPP", x_sub_dopp)

def plot_sub_doppler(
    cropped: CroppedData, lin: Linearised, off_res: OffResonance
) -> None:
    for x, y in off_res.selection_points:
        plt.scatter(x, y, s=100, marker="x", zorder=2, color="red")
    plt.plot(lin.product, cropped.Cropped_Sub_Doppler, zorder=1)
    plt.xlabel("Frequency with Arbitrary Offset (MHz)")
    plt.ylabel("Photodiode Output (V)")
    plt.show()


def plot_off_resonance(cropped: CroppedData, lin: Linearised, off_res: OffResonance) -> None:
    freq_mhz = lin.product

    plt.plot(off_res.off_resonance_x, off_res.off_resonance_trace, "ok")
    plt.plot(freq_mhz, cropped.Cropped_Sub_Doppler)
    plt.plot(freq_mhz, poly5(freq_mhz, *off_res.coefficients))
    plt.xlabel("Frequency with Arbitrary Offset (MHz)")
    plt.ylabel("Signal (V)")
    plt.show()

    plt.plot(freq_mhz / 1e3, off_res.normalised)
    plt.xlabel(FREQ_LABEL)
    plt.ylabel("Transmission")
    plt.show()


def sub_dopp_absolute_freq(t: np.ndarray, sub_dopp: np.ndarray):
    plt.plot(range(len(t)), sub_dopp)

    plt.title("Select eight points for hyperfine regions")
    x_cuts = np.array(plt.ginput(8))

    plt.show(block=False)
    plt.close()

    params.save_param("ABSOLUTE_FREQ_CUTS", x_cuts)


def plot_absolute_freq_cuts(sub_doppler, cuts) -> None:
    """cuts: iterable of (data-point index, value) pairs."""
    for x, y in cuts:
        plt.scatter(x, y, s=100, marker="x", zorder=2, color="red")
        plt.vlines(x, 0, max(sub_doppler), linestyle="dashed", color="black")
    plt.plot(range(len(sub_doppler)), sub_doppler)
    plt.xlabel("Data Point #")
    plt.ylabel("Photodiode Output (V)")
    plt.show()

def select_sub_doppler_hyperfine_peaks(
    detuning: np.ndarray,
    sub_dopp: np.ndarray,
    x_cuts: np.ndarray,
    *,
    n_points: int = 6,
    n_regions: int = 4,
):
    """
    x_cuts must be a 1D array
    """
    kdetuning = detuning / 1e3

    peaks: list[list] = []

    for i in range(0, n_regions * 2, 2):
        a, b = int(x_cuts[i,0]), int(x_cuts[i+1,0])

        plt.plot(kdetuning[a:b], sub_dopp[a:b])
        print(r'Select either side of atomic resonance')

        plt.title(f"Select {n_points} of the hyperfine structure")
        x_points = np.array(plt.ginput(n_points))[:, 0]

        plt.show(block=False)
        plt.close()

        peaks.extend(x_points.tolist())

    params.save_param("SUB_DOPPLER_DETECTED_PEAKS", np.array(peaks))

def figure_10(
    cropped: CroppedData, lin: Linearised, off_res: OffResonance, sd_peaks: SubDopplerPeaks
) -> None:
    freq_mhz = lin.product
    freq_ghz = freq_mhz / 1e3
    slope, intercept = sd_peaks.sub_doppler_peak_coefficients

    fig = plt.figure(figsize=(9 * 1.5, 9 * 1.3 * 1.5), constrained_layout=True)
    gs = GridSpec(14, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax2 = fig.add_subplot(gs[0:3, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[3:12, :2])
    ax4 = fig.add_subplot(gs[12:14, :2], sharex=ax3)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax1.plot(
        off_res.off_resonance_x / 1e3,
        off_res.off_resonance_trace,
        "o",
        color="black",
        rasterized=True,
    )
    ax1.plot(freq_ghz, cropped.Cropped_Sub_Doppler, RED)
    ax2.plot(freq_ghz, cropped.Cropped_Sub_Doppler, RED, zorder=1)
    ax2.plot(
        freq_ghz,
        poly5(freq_mhz, *off_res.coefficients),
        "black",
        linestyle="dashed",
        zorder=2,
        lw=2,
        rasterized=True,
    )
    ax3.plot(freq_ghz, off_res.normalised, color=RED)
    ax4.scatter(
        sd_peaks.detected_sub_doppler_peaks,
        sd_peaks.absolute_detunings,
        marker="x",
        s=4,
        color="black",
        zorder=2,
        lw=15,
    )
    ax4.plot(freq_ghz, slope * freq_ghz + intercept, OLIVE, zorder=2)

    ax4.set_xlim(0, 10)
    ax4.set_ylim(-5, 6)
    ax4.set_yticks([-4, 0, 4])
    ax3.set_ylim(0.16, 1.03)

    ax1.set_ylabel("Photodiode\n Output (V)", fontsize=22)
    for ax in (ax1, ax2, ax4):
        ax.set_xlabel(FREQ_LABEL, fontsize=22)
    ax4.set_ylabel("Linear Detuning\n (GHz)", fontsize=22)
    ax3.set_ylabel("Transmission with Zero Offset", fontsize=22)
    for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(axis="both", which="major", labelsize=22)
    plt.show()


def plot_smoothed_peaks(
    lin: Linearised, smoothed: SmoothedSubDoppler, sd_peaks: SubDopplerPeaks
) -> None:
    freq_ghz = lin.product / 1e3
    peaks = sd_peaks.detected_sub_doppler_peaks

    # (smoothed trace, x offset, title, xlim, ylim, yticks/xticks, peak slice)
    panels = [
        (
            smoothed.smooth100,
            0.005,
            "1st Absorption Peak",
            (1.4, 2),
            (0.78, 0.92),
            {"yticks": [0.8, 0.85, 0.9]},
            slice(0, 6),
        ),
        (
            smoothed.smooth100,
            0.005,
            "2nd Absorption Peak",
            (2.78, 3.1),
            (0.56, 0.72),
            {"yticks": [0.6, 0.65, 0.7]},
            slice(6, 12),
        ),
        (
            smoothed.smooth100,
            0.005,
            "3rd Absorption Peak",
            (5.78, 5.96),
            (0.70, 0.84),
            {"yticks": [0.72, 0.77, 0.82]},
            slice(12, 18),
        ),
        (
            smoothed.smooth70,
            0.0,
            "4th Absorption Peak",
            (8.15, 8.55),
            (0.855, 0.89),
            {"xticks": [8.3, 8.5]},
            slice(18, 24),
        ),
    ]

    _fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (trace, dx, title, xlim, ylim, ticks, sl) in zip(axs.flat, panels):
        ax.plot(freq_ghz + dx, trace, RED)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if "yticks" in ticks:
            ax.set_yticks(ticks["yticks"])
        if "xticks" in ticks:
            ax.set_xticks(ticks["xticks"])
        ax.set_title(title)
        ax.tick_params(axis="both", labelsize=12)
        for p in peaks[sl]:
            ax.vlines(p, ylim[0] - 0.01, ylim[1] + 0.01, linestyle="dashed", color="black", lw=0.7)
    plt.show()


# --------------------------------------------------------------------------- #
# Doppler-broadened spectrum normalisation
# --------------------------------------------------------------------------- #
def plot_doppler_selection(lin: Linearised, nd: NormalisedDoppler) -> None:
    for x, y in nd.selection_points:
        plt.scatter(x, y, s=100, marker="x", zorder=2, color="red")
    plt.plot(lin.product, nd.zeroed, zorder=1)
    plt.xlabel("Frequency with Arbitrary Offset (MHz)")
    plt.ylabel("Signal (V)")
    plt.show()


def plot_doppler_fit(lin: Linearised, nd: NormalisedDoppler) -> None:
    freq_mhz = lin.product

    plt.plot(nd.off_resonance_x, nd.off_resonance_trace, "ok")
    plt.plot(freq_mhz, nd.zeroed)
    plt.xlabel("Frequency with Arbitrary Offset (MHz)")
    plt.ylabel("Signal (V)")
    plt.show()

    plt.plot(freq_mhz / 1e3, nd.zeroed)
    plt.plot(
        freq_mhz / 1e3,
        poly5(freq_mhz, *nd.coefficients),
        color="black",
        zorder=2,
        linestyle="dashed",
    )
    plt.xlabel(FREQ_LABEL)
    plt.tick_params(axis="both", which="major", labelsize=22)
    plt.xticks([0, 5, 10])
    plt.show()

def select_doppler_absorption(detuning: np.ndarray, doppler: np.ndarray):
    plt.plot(detuning, doppler)

    print(r'Select either side of atomic resonance')
    x_dopp = plt.ginput(6)

    plt.show(block=False)
    plt.close()

    params.save_param("X_DOPP_BROAD", np.array(x_dopp))

def plot_normalised_doppler(lin: Linearised, nd: NormalisedDoppler, detuning_offset: float) -> None:
    """detuning_offset: intercept of the sub-Doppler absolute-frequency fit (GHz)."""
    plt.plot(lin.product / 1e3 + detuning_offset, nd.normalised)
    plt.ylabel("Transmission")
    plt.xlabel("Linear Detuning (GHz)")
    plt.show()

def select_hot_spectrum_zeros(hs_freq: np.ndarray, hs_volt: np.ndarray):
    plt.plot(hs_freq, hs_volt)

    print(r"Select either side of the 3 fully absorbed (zero light) regions")
    points = np.array(plt.ginput(6))

    plt.show(block=False)
    plt.close()

    params.save_param("HOT_SPECTRUM_ZEROS", points)

# --------------------------------------------------------------------------- #
# ElecSus theory fits
# --------------------------------------------------------------------------- #
def plot_theory_fit(fit: TheoryFit) -> None:
    _fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(fit.detuning, fit.values, "k.", label="Data")
    ax[0].plot(fit.detuning, fit.best_fit, "r-", lw=2, label="Fit")
    ax[0].set_ylabel("Transmission (S0)")
    ax[1].scatter(fit.detuning, fit.residuals)
    ax[1].set_xlabel("Linear Detuning (GHz)")
    plt.show()


def figure_13(fit20: TheoryFit, fit40: TheoryFit, fit73: TheoryFit) -> None:
    fig = plt.figure(figsize=(9 * 1.5, 9 * 1.6), constrained_layout=True)
    gs = GridSpec(14, 2, figure=fig)
    ax3 = fig.add_subplot(gs[1:8, :2])
    ax4 = fig.add_subplot(gs[8:10, :2], sharex=ax3)
    ax5 = fig.add_subplot(gs[10:12, :2], sharex=ax3)
    ax6 = fig.add_subplot(gs[12:14, :2], sharex=ax3)
    for ax in (ax3, ax4, ax5):
        plt.setp(ax.get_xticklabels(), visible=False)

    for fit, colour in ((fit20, PURPLE), (fit40, BLUE), (fit73, RED)):
        ax3.scatter(fit.detuning / 1e3, fit.values, s=5, color=colour)
    ax3.plot(fit20.detuning[20:] / 1e3, fit20.best_fit[20:], "black", linestyle="dashed", zorder=2)
    ax3.plot(fit40.detuning / 1e3, fit40.best_fit, "black", linestyle="dashed", zorder=2)
    ax3.plot(fit73.detuning / 1e3, fit73.best_fit, "black", linestyle="dashed", zorder=2)

    ax4.scatter(fit20.detuning / 1e3, fit20.residuals)
    ax5.scatter(fit40.detuning / 1e3, fit40.residuals, color=BLUE)
    ax6.scatter(fit73.detuning / 1e3, fit73.residuals, color=RED)

    ax5.set_yticks([-2, 0, 2])
    for ax in (ax3, ax4, ax5, ax6):
        ax.tick_params(axis="both", which="major", labelsize=22)
    ax6.set_xlabel("Linear Detuning (GHz)")
    ax5.set_ylabel("Residuals\n (x100)")
    ax3.set_ylabel("Normalised Transmission")
    plt.show()
