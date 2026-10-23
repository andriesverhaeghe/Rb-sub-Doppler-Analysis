from enum import Enum

from rb_spec_analysis.analysis import RbAnalysis
from rb_spec_analysis.plotting.paper import (
    figure_8,
    figure_10,
    plot_absolute_freq_cuts,
    plot_etalon_peaks,
    plot_off_resonance,
    plot_peak_difference,
    plot_peak_positions,
    plot_raw_signals,
    plot_smoothed_peaks,
    plot_sub_doppler,
    select_sub_doppler_hyperfine_peaks,
    sub_dopp_absolute_freq,
    sub_doppler_selection,
)


class Constants(Enum):
    FSR = 745  # MHz


class RbAnalysisSubDopp(RbAnalysis):
    def __init__(self, use_example_data: bool = False):
        super().__init__(use_example_data)

        self.constants = Constants

    def pipeline(self, select_points: bool = False, with_theory: bool = False) -> None:
        self.check_params()

        if with_theory:
            raise ValueError("The sub-Doppler pipeline does not support theory fits")

        if select_points:
            plot_raw_signals(
                self.Time, self.Normalised_Etalon, self.Sub_Doppler, doppler=self.Doppler
            )

        cropped_data = self.crop_frequency()

        lin = self.linearise_time(cropped_data)
        plot_etalon_peaks(cropped_data, lin)
        plot_peak_positions(lin)
        plot_peak_difference(cropped_data, lin)
        figure_8(cropped_data, lin)

        if select_points:
            sub_doppler_selection(lin.product, cropped_data.Cropped_Sub_Doppler)

        off_res = self.find_off_resonance(cropped_data, lin)
        plot_sub_doppler(cropped_data, lin, off_res)
        plot_off_resonance(cropped_data, lin, off_res)

        if select_points:
            sub_dopp_absolute_freq(lin.zeroed_linear_time, cropped_data.Cropped_Sub_Doppler)

        plot_absolute_freq_cuts(cropped_data.Cropped_Sub_Doppler, self._load("ABSOLUTE_FREQ_CUTS"))

        if select_points:
            select_sub_doppler_hyperfine_peaks(
                lin.product, cropped_data.Cropped_Sub_Doppler, self._load("ABSOLUTE_FREQ_CUTS")
            )

        sd_peaks = self.sd_peaks()
        figure_10(cropped_data, lin, off_res, sd_peaks)
        plot_smoothed_peaks(lin, self.smooth(lin, off_res, cropped_data), sd_peaks)

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

    RbAnalysisSubDopp(use_example_data=args.example_data).pipeline(
        select_points=args.select_points, with_theory=args.theory
    )
