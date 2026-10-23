"""Rb spectroscopy analysis: all fitting/processing, plus the entry points.

This module owns the pipeline. It computes results, packs them into the
schemas in ``schemas/analysis_schemas.py`` and hands them to ``plotting.py``,
which only draws. ``plotting.py`` must not import from this module.
"""

from enum import Enum
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import detrend, find_peaks
from scipy.stats import binned_statistic

from rb_spec_analysis.plotting.paper import (
    figure_8,
    figure_10,
    figure_13,
    plot_absolute_freq_cuts,
    plot_crop_selection,
    plot_cropped_signals,
    plot_doppler_fit,
    plot_doppler_selection,
    plot_etalon_peaks,
    plot_linearised_etalon,
    plot_normalised_doppler,
    plot_off_resonance,
    plot_peak_difference,
    plot_peak_positions,
    plot_raw_signals,
    plot_smoothed_peaks,
    plot_sub_doppler,
    plot_theory_fit,
    select_hot_spectrum_zeros,
    select_sub_doppler_hyperfine_peaks,
    sub_dopp_absolute_freq,
    sub_doppler_selection,
)
from rb_spec_analysis.schemas.analysis_schemas import (
    Linearised,
    NormalisedDoppler,
    OffResonance,
    SmoothedSubDoppler,
    SubDopplerPeaks,
    TheoryFit,
)
from rb_spec_analysis.utils.array import zero
from rb_spec_analysis.utils.data import CroppedData, Data, load_data
from rb_spec_analysis.utils.params import Params
from rb_spec_analysis.utils.polynomials import poly1, poly5, poly7


class Constants(Enum):
    FSR = .85714  # GHz


class RbAnalysis:
    def __init__(self, use_example_data: bool = False):
        # this file: <repo>/src/rb_spec_analysis/analysis.py -> parents[2] is <repo>
        root = Path(__file__).resolve().parents[2]
        dataloc: Path = root / ("example_data" if use_example_data else "real_data")

        self.prm_loc = Path(__file__).resolve().parent / "params"

        self.data: Data = load_data(dataloc)
        self.__preprocess()

        self.constants = Constants
        self.params = Params()

        self.load_param_retries = 0

    def __preprocess(self) -> None:
        d = self.data

        self.Time = d.etalon_signal[:, 0]
        self.Etalon = d.etalon_signal[:, 1]
        self.Sub_Doppler = d.sub_doppler_signal[:, 1]

        if not np.allclose(self.Time, d.sub_doppler_signal[:, 0]):
            raise ValueError("etalon and sub-Doppler time bases differ")

        self.Normalised_Etalon = self.Etalon / self.Etalon.max()

        # Optional Doppler data
        self.Doppler = self.Normalised_Doppler = None
        if d.rb_spectrum_signal_20C is not None:
            _, self.Doppler = d.rb_spectrum_signal_20C.T
            self.Normalised_Doppler = self.Doppler / self.Doppler.max()

        self.Doppler_2_Freq = self.Doppler_2_Trnsm = None
        if d.rb_spectrum_signal_40C is not None:
            self.Doppler_2_Freq, self.Doppler_2_Trnsm = d.rb_spectrum_signal_40C.T  # MHz, T

        self.Doppler_3_Freq = self.Doppler_3_Trnsm = None
        if d.rb_spectrum_signal_73C is not None:
            self.Doppler_3_Freq, self.Doppler_3_Trnsm = d.rb_spectrum_signal_73C.T  # MHz, T

        self.HS_Freq = self.HS_Volt = None
        if d.hot_spectrum_signal is not None:
            self.HS_Freq, self.HS_Volt = d.hot_spectrum_signal.T

    @staticmethod
    def __smoothTriangle(data, degree):
        triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
        smoothed = []

        for i in range(degree, len(data) - degree * 2):
            point = data[i : i + len(triangle)] * triangle
            smoothed.append(np.sum(point) / np.sum(triangle))

        smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return np.array(smoothed)

    @staticmethod
    def __with_crossovers(lines) -> np.ndarray:
        """Return the lines plus the midpoint of every pair (crossover resonances)."""
        lines = np.asarray(lines, dtype=float)
        i, j = np.triu_indices(len(lines), k=1)
        return np.concatenate([lines, (lines[i] + lines[j]) / 2])

    @staticmethod
    def __off_resonance_mask(x: np.ndarray, selection_points) -> np.ndarray:
        """True everywhere except between points (0,1), (2,3) and (4,5),
        which bracket the features."""
        a, b, c, d, e, f = (int(np.argmin(np.abs(x - p[0]))) for p in selection_points)
        mask = np.ones(x.shape, dtype=bool)
        for lo, hi in ((a, b), (c, d), (e, f)):
            mask[lo:hi] = False
        return mask

    def _load(self, file_name: str) -> np.ndarray:
        try:
            return np.load(self.prm_loc / (file_name + ".npy"))

        except OSError:
            self.load_param_retries += 1

            if self.load_param_retries < 3:
                self.params.reset_param(file_name)
                param = self._load(file_name)

                self.load_param_retries = 0
                return param
            else:
                return np.ndarray([])

        except (ValueError, EOFError) as e:
            print(e)
            return np.ndarray([])

        except Exception as e:
            print(e)
            raise

    def check_params(self):
        if not (self.prm_loc.is_dir() and any(self.prm_loc.iterdir())):
            print("No Params found, loading from defaults")
            self.params.reset()

    def crop_frequency(self) -> CroppedData:
        t_min, t_max = self.Time.min(), self.Time.max()
        n = len(self.Time)

        X_CROP = self._load("X_CROP")

        start = int((X_CROP[0] - t_min) * n / (t_max - t_min))
        finish = int((X_CROP[1] - t_min) * n / (t_max - t_min))

        etalon = self.Etalon[start:finish]
        doppler = self.Doppler

        return CroppedData(
            Cropped_Time=zero(self.Time[start:finish]),
            # The notebook normalises *after* cropping, so peak = 1 within the cropped region.
            Cropped_Etalon=etalon / etalon.max(),
            Cropped_Doppler=doppler[start:finish] if doppler is not None else None,
            Cropped_Sub_Doppler=self.Sub_Doppler[start:finish],
        )

    def linearise_time(self, cropped_data: CroppedData) -> Linearised:
        normalised_etalon = cropped_data.Cropped_Etalon
        zeroed_time = cropped_data.Cropped_Time

        peak_idx, _ = find_peaks(detrend(normalised_etalon), height=0.0, distance=100)
        peak_idx = np.asarray(peak_idx, dtype=int)

        etalon_peaks = normalised_etalon[peak_idx]
        peak_time_ms = zeroed_time[peak_idx]
        peak_number = np.arange(len(etalon_peaks))

        # Linear fit: peak number -> peak time
        peak_coeffs, _ = curve_fit(poly1, peak_number, peak_time_ms, [1.0] * 2)

        # Deviation from linear, fitted with a 7th order polynomial in time
        difference = peak_time_ms - poly1(peak_number, *peak_coeffs)
        difference_coeffs, _ = curve_fit(poly1, peak_time_ms, difference, [1.0] * 2)

        linearised_time = zeroed_time - poly1(zeroed_time, *difference_coeffs)
        zeroed_linear_time = zero(linearised_time)

        zeroed_time_peaks = zeroed_linear_time[peak_idx]
        peak_spacing = float(
            (zeroed_time_peaks[-1] - zeroed_time_peaks[0]) / (len(zeroed_time_peaks) - 1)
        )
        freq_per_time = self.constants.FSR.value / peak_spacing  # GHz per ms

        return Linearised(
            zeroed_linear_time=zeroed_linear_time,
            freq_per_time=freq_per_time,
            peak_idx=peak_idx,
            peak_number=peak_number,
            peak_time_ms=peak_time_ms,
            etalon_peaks=etalon_peaks,
            peak_coeffs=peak_coeffs,
            difference=difference,
            difference_coeffs=difference_coeffs,
        )

    def find_off_resonance(self, cropped_data: CroppedData, linearised: Linearised) -> OffResonance:
        x = linearised.product  # MHz
        sub_doppler = cropped_data.Cropped_Sub_Doppler

        X_SUB_DOPP = self._load("X_SUB_DOPP")

        mask = self.__off_resonance_mask(x, X_SUB_DOPP)
        off_res_x = x[mask]
        off_res_trace = sub_doppler[mask]

        coeffs, _ = curve_fit(poly5, off_res_x, off_res_trace, [1.0] * 6)

        return OffResonance(
            off_resonance_x=off_res_x,
            off_resonance_trace=off_res_trace,
            coefficients=coeffs,
            selection_points=list(X_SUB_DOPP),
            normalised=sub_doppler / poly5(x, *coeffs),
        )

    def sd_peaks(self) -> SubDopplerPeaks:
        """Linear fit of detected sub-Doppler peak positions (GHz) to absolute detunings (GHz)."""
        absolute_detunings = np.sort(
            np.concatenate([self.__with_crossovers(g) for g in self._load("HYPERFINE_GROUPS")])
        )
        detected_peaks = np.sort(np.asarray(self._load("SUB_DOPPLER_DETECTED_PEAKS"), dtype=float))

        coeffs, _ = curve_fit(poly1, detected_peaks, absolute_detunings, [1.0] * 2)

        return SubDopplerPeaks(
            sub_doppler_peak_coefficients=coeffs,
            detected_sub_doppler_peaks=detected_peaks,
            absolute_detunings=absolute_detunings,
        )

    def smooth(
        self,
        linearised: Linearised,
        off_resonance: OffResonance,
        cropped_data: CroppedData,
    ) -> SmoothedSubDoppler:
        sub_doppler = cropped_data.Cropped_Sub_Doppler

        # NOTE: this reproduces the notebook exactly. The poly5 was fitted against MHz but is
        # evaluated here at GHz (x/1e3), so this is not a true background division. The plotting
        # axis limits were chosen against this scaling, and the smoothed data is only used to
        # pick peaks by eye, so it is kept as-is. Pass `linearised.product` instead to fix it.
        background = poly5(linearised.product / 1e3, *off_resonance.coefficients)

        return SmoothedSubDoppler(
            smooth100=self.__smoothTriangle(sub_doppler, 100) / background,
            smooth70=self.__smoothTriangle(sub_doppler, 70) / background,
        )

    def normalise_doppler(
        self, linearised: Linearised, cropped_data: CroppedData
    ) -> NormalisedDoppler:
        doppler = cropped_data.Cropped_Doppler
        if doppler is None:
            raise ValueError("Doppler-broadened (20C) data is required to normalise the Doppler")

        x = linearised.product  # MHz

        DOPPLER_SLOPE_INT = self._load("DOPPLER_SLOPE_INT")
        X_DOPP_BROAD = self._load("X_DOPP_BROAD")

        zero_level = (
            DOPPLER_SLOPE_INT[0] * linearised.zeroed_linear_time / 1e3 * linearised.freq_per_time
            + DOPPLER_SLOPE_INT[1]
        )
        zeroed = doppler - zero_level

        mask = self.__off_resonance_mask(x, X_DOPP_BROAD)
        off_res_x = x[mask]
        off_res_trace = zeroed[mask]

        coeffs, _ = curve_fit(poly5, off_res_x, off_res_trace, [1.0] * 6)

        return NormalisedDoppler(
            zeroed=zeroed,
            selection_points=list(X_DOPP_BROAD),
            off_resonance_x=off_res_x,
            off_resonance_trace=off_res_trace,
            coefficients=coeffs,
            normalised=zeroed / poly5(x, *coeffs),
        )

    def fit_doppler_zero(self) -> np.ndarray:
        """Linear fit (V/GHz, V) to the zero level of the hot spectrum."""
        if self.HS_Freq is None or self.HS_Volt is None:
            raise ValueError("Hot spectrum data (Dopp_Broadened_73C Lin.csv) is required")

        points = self._load("HOT_SPECTRUM_ZEROS")
        zero_mask = ~self.__off_resonance_mask(self.HS_Freq, points)

        coeffs, _ = curve_fit(
            poly1, self.HS_Freq[zero_mask], self.HS_Volt[zero_mask], [1.0] * 2
        )
        return coeffs

    @staticmethod
    def __bin(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
        """Mean-bin y and x onto `bins` equal-width bins of x."""
        y_binned = binned_statistic(x, y, statistic="mean", bins=bins)[0]
        x_binned = binned_statistic(x, x, statistic="mean", bins=bins)[0]
        return x_binned, y_binned

    @staticmethod
    def __fit_elecsus(
        detuning: np.ndarray, values: np.ndarray, t_guess: float, t_bounds: tuple[float, float]
    ) -> TheoryFit:
        # Imported here so the non-theory entry points work without loading ElecSus fitting
        import elecsus.elecsus_methods as EM

        e_in = np.array([1, 0, 0])  # linear horizontal
        e_in_angle = [e_in[0].real, [abs(e_in[1]), np.angle(e_in[1])]]

        p_dict_guess = {
            "Elem": "Rb",
            "Dline": "D2",
            "T": t_guess,
            "lcell": 75e-3,
            "Btheta": 0,
            "Bfield": 0,
            "shift": 0,
            "Bphi": 0,
            "GammaBuf": 0,
        }
        p_dict_bools = {"T": True}
        p_dict_bounds = {"T": list(t_bounds)}

        _best_params, _rms, result = EM.fit_data(
            [detuning, values],
            p_dict_guess,
            p_dict_bools,
            e_in_angle,
            p_dict_bounds,
            data_type="S0",
            fit_algorithm="ML",
        )
        return TheoryFit(detuning=detuning, values=values, best_fit=result.best_fit)

    def fit_theory(
        self, linearised: Linearised, normalised: NormalisedDoppler, detuning_offset: float
    ) -> tuple[TheoryFit, TheoryFit, TheoryFit]:
        """ElecSus temperature fits of the 20C, 40C and 73C spectra.

        detuning_offset: intercept of the sub-Doppler absolute-frequency fit (GHz).
        """
        freq_40, trnsm_40 = self.Doppler_2_Freq, self.Doppler_2_Trnsm
        freq_73, trnsm_73 = self.Doppler_3_Freq, self.Doppler_3_Trnsm
        if freq_40 is None or trnsm_40 is None:
            raise ValueError("40C Doppler-broadened data is required for the theory fits")
        if freq_73 is None or trnsm_73 is None:
            raise ValueError("73C Doppler-broadened data is required for the theory fits")

        THEORY_BINS = 800

        x20 = linearised.product + detuning_offset * 1e3  # MHz
        detuning_20, values_20 = self.__bin(x20, normalised.normalised, THEORY_BINS)
        detuning_40, values_40 = self.__bin(freq_40, trnsm_40, THEORY_BINS)
        detuning_73, values_73 = self.__bin(freq_73, trnsm_73, THEORY_BINS)

        # ElecSus fits: (temperature guess C, temperature bounds C)
        t = {
            "20C": (18, (0, 25)),
            "40C": (45, (30, 50)),
            "73C": (75, (60, 85)),
        }

        return (
            self.__fit_elecsus(detuning_20, values_20, *t["20C"]),
            self.__fit_elecsus(detuning_40, values_40, *t["40C"]),
            self.__fit_elecsus(detuning_73, values_73, *t["73C"]),
        )

    def pipeline(self, select_points: bool = False, with_theory: bool = False) -> None:
        self.check_params()
        
        if select_points:
            plot_raw_signals(
                self.Time, self.Normalised_Etalon, self.Sub_Doppler, doppler=self.Doppler
            )
        
        cropped_data = self.crop_frequency()
        plot_crop_selection(
            self.Time,
            self.Normalised_Etalon,
            self.Normalised_Doppler,
            self.Sub_Doppler,
            self._load("X_CROP")
        )
        plot_cropped_signals(cropped_data)

        lin = self.linearise_time(cropped_data)
        plot_etalon_peaks(cropped_data, lin)
        plot_peak_positions(lin)
        plot_peak_difference(cropped_data, lin)
        plot_linearised_etalon(cropped_data, lin)
        figure_8(cropped_data, lin)

        if select_points:
            sub_doppler_selection(lin.product, cropped_data.Cropped_Sub_Doppler)

        off_res = self.find_off_resonance(cropped_data, lin)
        plot_sub_doppler(cropped_data, lin, off_res)
        plot_off_resonance(cropped_data, lin, off_res)

        if select_points:
            sub_dopp_absolute_freq(lin.product, cropped_data.Cropped_Sub_Doppler)

        plot_absolute_freq_cuts(lin.product, cropped_data.Cropped_Sub_Doppler, self._load("ABSOLUTE_FREQ_CUTS"))

        if select_points:
            select_sub_doppler_hyperfine_peaks(
                lin.product, cropped_data.Cropped_Sub_Doppler, self._load("ABSOLUTE_FREQ_CUTS")
            )

        sd_peaks = self.sd_peaks()
        figure_10(cropped_data, lin, off_res, sd_peaks)
        # plot_smoothed_peaks(lin, self.smooth(lin, off_res, cropped_data), sd_peaks)
        ranges = lin.product[self._load("ABSOLUTE_FREQ_CUTS")]
        plot_smoothed_peaks(lin, off_res, sd_peaks, ranges)

        if select_points and self.HS_Freq is not None and self.HS_Volt is not None:
            select_hot_spectrum_zeros(self.HS_Freq, self.HS_Volt)
            self.params.save_param("DOPPLER_SLOPE_INT", self.fit_doppler_zero())

        detuning_offset = float(sd_peaks.sub_doppler_peak_coefficients[1])  # GHz
        nd = self.normalise_doppler(lin, cropped_data)
        plot_doppler_selection(lin, nd)
        plot_doppler_fit(lin, nd)
        plot_normalised_doppler(lin, nd, detuning_offset)

        if with_theory:
            fits = self.fit_theory(lin, nd, detuning_offset)
            for fit in fits:
                plot_theory_fit(fit)
            figure_13(*fits)


# --------------------------------------------------------------------------- #
# Entrypoint (see [project.scripts] in pyproject.toml)
# --------------------------------------------------------------------------- #

import argparse


def pipeline() -> None:
    parser = argparse.ArgumentParser(description="Run the Rb spectrum analysis pipeline.")

    parser.add_argument("--theory", action="store_true", help="Include the theory comparison.")
    parser.add_argument(
        "--example-data",
        action="store_true",
        help="Use real data instead of the bundled example data.",
    )
    parser.add_argument(
        "--select-points",
        action="store_true",
        help="Manually chose plotting points such as cropping range, absorption peaks etc.",
    )

    args = parser.parse_args()

    RbAnalysis(use_example_data=args.example_data).pipeline(
        select_points=args.select_points, with_theory=args.theory
    )
