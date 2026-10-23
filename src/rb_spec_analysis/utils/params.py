"""
Default values taken from paper using pyplot.ginput
"""

import shutil
from pathlib import Path
from typing import ClassVar

import numpy as np


class Params:
    params_loc: ClassVar[Path] = Path(__file__).resolve().parents[1] / "params"

    defaults_dict: ClassVar[dict[str, np.ndarray]] = {
        "X_CROP": np.array([-0.064253938172043, 0.07161903837365591]),

        # Either side of the sub-Doppler absorption features, (MHz, V)
        "X_SUB_DOPP": np.array([
            [1035.4632403415858, 0.56511347],
            [3712.0629829359273, 0.56778205],
            [5137.643280622044, 0.56968818],
            [6570.496947173906, 0.56473224],
            [7726.9625968274395, 0.55367668],
            [9094.355943587592, 0.53728394],
        ]),

        # Either side of the Doppler-broadened features, (MHz, V)
        "X_DOPP_BROAD": np.array([
            [1064.556716627084, 0.24846527746397118],
            [3748.4298301608305, 0.24696917181031855],
            [5137.643284591577, 0.24761035994759825],
            [6592.31705886461, 0.24525933677757267],
            [7719.68923392621, 0.23884745540477562],
            [9116.176057228324, 0.23051200962013946],
        ]),

            # (data-point index, value) pairs bracketing each absorption region; display only
        "ABSOLUTE_FREQ_CUTS": np.array([
            [19221.139784946237, 0.5327657131482988],
            [27859.23870967742, 0.46720791497669234],
            [38204.8688172043, 0.3859315804013672],
            [43427.90537634408, 0.34337651843032435],
            [79888.7182795699, 0.4162185163987761],
            [83002.45161290324, 0.4204356847022128],
            [112934.46881720432, 0.496344714164073],
            [118057.06236559141, 0.50247877715089],
        ]),

        "SUB_DOPPLER_DETECTED_PEAKS": np.array([
            1.8780, 1.7460, 1.6687, 1.6118,
            1.5344, 1.4542, 2.8238, 2.8533,
            2.8842, 2.9150, 2.9436, 3.0084,
            8.4416, 8.3637, 8.3376, 8.2853,
            8.2499, 8.2172, 5.8740, 5.9057,
            5.8423, 5.8130, 5.8277, 5.8594,
        ]),

        # Absolute hyperfine line positions (GHz), Siddons et al. 2008
        "HYPERFINE_GROUPS": np.array([
            [-2.73505, -2.57811, -2.31126],
            [-1.37129, -1.30787, -1.18691],
            [1.635454, 1.664714, 1.728134],
            [4.027403, 4.099625, 4.2565],
        ]),

        # Zero offset of the Doppler-broadened trace, fitted from the hot spectrum
        "DOPPLER_SLOPE_INT": np.array([
            -0.00046452752822592503,  # V per GHz
            0.05549039777557824,  # V
        ]),

        "HOT_SPECTRUM_ZEROS": np.array([
            [1.7400576705241937, 0.05543383919665015],
            [3.9135362992741944, 0.053847781771920115],
            [5.806566072701615, 0.053847781771920115],
            [6.8270882962365596, 0.051733038538946736],
            [8.416298476397852, 0.051204352730703395],
            [9.17195233657258, 0.05067566692246005],
        ]),
    }

    @classmethod
    def reset(cls):
        cls.params_loc.mkdir(parents=True, exist_ok=True)

        for key, value in cls.defaults_dict.items():
            np.save(cls.params_loc / key, value)

    def reset_param(self, param_name: str):
        default_val = self.defaults_dict.get(param_name)

        if default_val is not None:
            np.save(self.params_loc / param_name, default_val)
        else:
            print(f"No default value found for {param_name}")

    def save_param(self, param_name: str, param_val: np.ndarray):
        np.save(self.params_loc / (param_name + ".npy"), param_val)

    def save_new_params(self):
        new_params_loc = self.params_loc.parent / "saved_params"
        shutil.copytree(self.params_loc, new_params_loc, dirs_exist_ok=True)

    def pull_params(self):
        pulled_params_loc = self.params_loc.parent / "saved_params"
        if pulled_params_loc.exists():
            shutil.copytree(pulled_params_loc, self.params_loc)


def reset_defaults():
    params = Params()
    params.reset()

def save_params():
    params = Params()
    params.save_new_params()

def pull_params():
    params = Params()
    params.pull_params()
