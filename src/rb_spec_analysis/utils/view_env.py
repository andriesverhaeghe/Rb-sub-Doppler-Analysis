import sys

import elecsus
import lmfit
import matplotlib
import numpy as np
import scipy
import sympy
import wx


def print_packages() -> None:
    print(f"Python version: {sys.version}")
    print(f"numpy Version: {np.__version__}")
    print(f"matplotlib Version: {matplotlib.__version__}")
    print(f"sympy Version: {sympy.__version__}")
    print(f"scipy Version: {scipy.__version__}")
    print(f"lmfit Version: {lmfit.__version__}")
    try:
        print(f"wx Version: {wx.__version__}")
        print(f"elecus Loc: {elecsus.__file__}")
    except (ImportError, AttributeError):
        print("Core dependencies installed, no wxPython and/or ElecSus")
