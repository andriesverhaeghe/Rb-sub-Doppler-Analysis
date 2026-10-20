from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Data:
    # required: the core etalon + sub-Doppler pipeline can't run without these
    etalon_signal: np.ndarray
    sub_doppler_signal: np.ndarray

    # optional: only the Doppler / theory steps need these
    rb_spectrum_signal_20C: np.ndarray | None = None
    rb_spectrum_signal_40C: np.ndarray | None = None
    rb_spectrum_signal_73C: np.ndarray | None = None
    hot_spectrum_signal: np.ndarray | None = None


@dataclass
class CroppedData:
    Cropped_Time: np.ndarray
    Cropped_Etalon: np.ndarray
    Cropped_Sub_Doppler: np.ndarray

    Cropped_Doppler: np.ndarray | None = None


def _load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype="float", skiprows=1)


def _load_optional(path: Path) -> np.ndarray | None:
    return _load_csv(path) if path.is_file() else None


def load_data(loc: Path) -> Data:
    return Data(
        etalon_signal=_load_csv(loc / "Etalon.csv"),
        sub_doppler_signal=_load_csv(loc / "Sub_Doppler Spectrum.csv"),
        rb_spectrum_signal_20C=_load_optional(loc / "Dopp_Broadened_20C.csv"),
        rb_spectrum_signal_40C=_load_optional(loc / "Dopp_Broadened_40C Norm.csv"),
        rb_spectrum_signal_73C=_load_optional(loc / "Dopp_Broadened_73C Norm.csv"),
        hot_spectrum_signal=_load_optional(loc / "Dopp_Broadened_73C Lin.csv"),
    )
