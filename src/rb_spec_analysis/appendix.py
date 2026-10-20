from typing import ClassVar

import numpy as np
from elecsus.libs import spectra as sp
from scipy.optimize import curve_fit

from rb_spec_analysis.analysis import (
    RbAnalysis,
)
from rb_spec_analysis.plotting.appendix import (
    basic_1var_signal_time,
    basic_2var_signal_time,
    figure_9,
    plot_x_points,
    scatter_and_plot,
)
from rb_spec_analysis.utils.polynomials import (
    poly1,
    poly5,
)


class Appendix(RbAnalysis):
    """
    Appendix section of notebook as a class
    """

    x_zeros: ClassVar[list[tuple]] = [
        (1.7400576705241937, 0.05543383919665015),
        (3.9135362992741944, 0.053847781771920115),
        (5.806566072701615, 0.053847781771920115),
        (6.8270882962365596, 0.051733038538946736),
        (8.416298476397852, 0.051204352730703395),
        (9.17195233657258, 0.05067566692246005),
    ]

    x_ones: ClassVar[list[tuple]] = [
        (0.5637305273655917, 2.921890603386382),
        (4.747092619260753, 2.837981256157135),
        (4.980799998696238, 2.847304516960385),
        (7.489259204637098, 2.791364952140887),
        (7.691805600147851, 2.8100114737473865),
        (10.208055052069895, 2.7773800609360126),
    ]

    def __init__(self, use_example_data: bool = False):
        super().__init__(use_example_data)

    def appendix(self):
        ## Appendix: Normalising and Deriving a Zero from a Hot Spectrum
        ## (Following 'Zero-ing' Discussion in Section 3.2)

        # The appendix needs the hot spectrum and the Doppler trace; fail early and clearly
        # if they weren't loaded. Narrowed into locals so the type checker knows they're arrays.
        hs_freq, hs_volt = self.HS_Freq, self.HS_Volt
        if hs_freq is None or hs_volt is None:
            raise ValueError("Hot spectrum data (Dopp_Broadened_73C Lin.csv) is required")

        cropped_data = self.crop_frequency()
        doppler = cropped_data.Cropped_Doppler
        if doppler is None:
            raise ValueError("Doppler-broadened (20C) data is required")

        linearised = self.linearise_time(cropped_data)

        """
        #Step 1: Find the Zero Line
        %matplotlib
        plt.plot(Hot_Spectrum_Frequencies,Hot_Spectrum_Voltages)


        #Collect mouse input
        x_zeros = plt.ginput(6) 

        plt.show(block=False)
        plt.close()
        """

        ## Zero Points from Paper

        plot_x_points(hs_freq, hs_volt, self.x_zeros)

        ## Zero Regions Defined
        a, b, c, d, e, f = [np.argmin(np.abs(hs_freq - self.x_zeros[i][0])) for i in range(6)]

        offResX = np.append(hs_freq[a:b], hs_freq[c:d])
        offResX = np.append(offResX, hs_freq[e:f])

        offResTrace = np.append(hs_volt[a:b], hs_volt[c:d])
        offResTrace = np.append(offResTrace, hs_volt[e:f])

        basic_1var_signal_time(offResX, offResTrace, "o")
        basic_2var_signal_time(offResX, offResTrace, hs_freq, hs_volt, "ok")

        ## Fit Zeros. Change polynomial order as needed.

        coeffs, _matcov = curve_fit(poly1, offResX, offResTrace, [1.0] * 2)

        # a) from Figure 9

        figure_9(hs_freq, hs_volt, coeffs)

        ## Hot Spectrum Zeroed

        Zeroed_Spec = hs_volt / poly1(hs_freq, *coeffs) - 1
        figure_9(hs_freq, Zeroed_Spec)

        basic_2var_signal_time(
            linearised.product / 1e3,
            doppler - poly1(linearised.product / 1e3, *coeffs),
            linearised.product / 1e3,
            doppler,
            xlabel="Frequency with Arbitrary Offset (GHz)",
            ylabel="Photodiode \n Output (V)",
        )

        ## We now continue linearisation of hot spectrum
        ## Choose points where spectrum is 1 or near 1
        ## One Points Used in Paper

        plot_x_points(hs_freq, Zeroed_Spec, self.x_ones)

        ## One regions defined. Note that three regions are not exactly one.
        ## The choice of offsets from one (0.02, 0.13 and 0.09 in this case)
        ## are found through trial and error.

        a, b, c, d, e, f = [np.argmin(np.abs(hs_freq - self.x_ones[i][0])) for i in range(6)]

        offResX = np.append(hs_freq[:a], hs_freq[b:c])
        offResX = np.append(offResX, hs_freq[d:e])
        offResX = np.append(offResX, hs_freq[f:])

        offResTrace = np.append((Zeroed_Spec[:a]) + 0.02, (Zeroed_Spec[b:c]) + 0.13)
        offResTrace = np.append(offResTrace, (Zeroed_Spec[d:e]) + 0.09)
        offResTrace = np.append(offResTrace, (Zeroed_Spec[f:]))

        scatter_and_plot(offResX, offResTrace, hs_freq, Zeroed_Spec)

        ## Fitting the ones with a polynomial

        coeffs, _matcov = curve_fit(poly5, offResX, offResTrace, [1.0] * 6)

        ## Normalised Hot Spectrum

        Norm_Hot_Spec = Zeroed_Spec / poly5(hs_freq, *coeffs)

        detuning = np.linspace(-4.5, 5.7, 1000) * 1e3
        E_in = np.array([1, 0, 0])

        p = {"Elem": "Rb", "Dline": "D2", "T": 72.7, "lcell": 75e-3}

        # Unpack fit
        fit, *_ = sp.get_spectra(detuning, E_in, p, outputs=["S0"])

        # -4.6502350438882285 from absolute frequency calibration
        basic_2var_signal_time(
            detuning / 1e3,
            fit,
            hs_freq - 4.6502350438882285,
            Norm_Hot_Spec,
            xlabel="Linear Detuning (GHz)",
            ylabel="Transmission",
        )
