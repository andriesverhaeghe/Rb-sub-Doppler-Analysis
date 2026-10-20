from typing import NamedTuple

import numpy as np


class Linearised(NamedTuple):
    zeroed_linear_time: np.ndarray
    freq_per_time: float
    peak_idx: np.ndarray
    peak_number: np.ndarray
    peak_time_ms: np.ndarray
    etalon_peaks: np.ndarray
    peak_coeffs: np.ndarray  # poly1 fit: peak number -> time
    difference: np.ndarray
    difference_coeffs: np.ndarray  # poly7 fit to difference

    @property
    def product(self) -> np.ndarray:
        return self.zeroed_linear_time * self.freq_per_time


class OffResonance(NamedTuple):
    off_resonance_x: np.ndarray
    off_resonance_trace: np.ndarray
    coefficients: np.ndarray
    selection_points: list[tuple[float, float]]
    normalised: np.ndarray  # sub-Doppler / poly5 background


class SubDopplerPeaks(NamedTuple):
    sub_doppler_peak_coefficients: np.ndarray
    detected_sub_doppler_peaks: np.ndarray
    absolute_detunings: np.ndarray


class SmoothedSubDoppler(NamedTuple):
    smooth100: np.ndarray  # already divided by the background fit
    smooth70: np.ndarray


class NormalisedDoppler(NamedTuple):
    zeroed: np.ndarray
    selection_points: list[tuple[float, float]]
    off_resonance_x: np.ndarray
    off_resonance_trace: np.ndarray
    coefficients: np.ndarray
    normalised: np.ndarray


class TheoryFit(NamedTuple):
    detuning: np.ndarray  # MHz
    values: np.ndarray
    best_fit: np.ndarray

    @property
    def residuals(self) -> np.ndarray:
        return 100 * (self.best_fit - self.values)
