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
        "X_CROP": np.array([-0.05124325, 0.06200887]),

        # Either side of the sub-Doppler absorption features, (MHz, V)
        "X_SUB_DOPP": np.array([
            [0.0863689,0.56813689],
            [3.18802847,0.56862988],
            [4.87857264,0.57010886],
            [6.48528819,0.56665791],
            [7.7986035,0.55384008],
            [9.34943328,0.53461333],
        ]),

        # Either side of the Doppler-broadened features, (GHz, V)
        "X_DOPP_BROAD": np.array([
            [0.21211185,0.30213785],
            [3.21597134,0.30050764],
            [4.9344584,0.30023594],
            [6.38748811,0.29833403],
            [7.85448926,0.29018299],
            [9.23766177,0.28176024]
        ]),

        # (data-point index, value) pairs bracketing each absorption region; display only
        "ABSOLUTE_FREQ_CUTS": np.array([
            [4886, 0.5327657131482988],
            [17779, 0.46720791497669234],
            [22467, 0.3859315804013672],
            [35863, 0.34337651843032435],
            [62486, 0.4162185163987761],
            [73035, 0.4204356847022128],
            [96812, 0.496344714164073],
            [109035, 0.50247877715089],
        ]),

        "SUB_DOPPLER_DETECTED_PEAKS": np.array(
            [0.7021056,0.77585291,0.85116932,0.94060755,
             0.60482191,1.084964,2.18102517,2.20743686,
             2.25035586,2.28337047,2.31638508,2.38901722,
             5.41565786,5.62829713,5.65061112,5.68998877,
             5.73199159,5.86981333,8.42077832,8.46588684,
             8.51851346,8.55610389,8.63729924,8.85231655]
        ),

        # Absolute hyperfine line positions (GHz), Siddons et al. 2008
        "HYPERFINE_GROUPS": np.array([
            [-2.73505, -2.57811, -2.31126],
            [-1.37129, -1.30787, -1.18691],
            [1.635454, 1.664714, 1.728134],
            [4.027403, 4.099625, 4.2565],
        ]),

        # Zero offset of the Doppler-broadened trace, fitted from the hot spectrum
        "DOPPLER_SLOPE_INT": np.array([
            0.00194638,  # V per GHz
            0.08301687,  # V
        ]),

        "HOT_SPECTRUM_ZEROS": np.array([
            [1.01556479, 0.21068181],
            [4.50559499, 0.20168708],
            [5.31578058, 0.20237898],
            [7.37240551, 0.19580591],
            [7.82423978, 0.19511401],
            [9.9276062,  0.19200045],
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
