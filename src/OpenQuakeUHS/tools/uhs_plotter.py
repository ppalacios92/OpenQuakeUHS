"""
UHS Plotting Utilities
Author: Ing. Patricio Palacios Msc.
Date: June 6, 2025

Description:
------------
This module defines a function to visualize Uniform Hazard Spectra (UHS)
from OpenQuake CSV outputs, plotting multiple realizations, quantiles,
and the mean curve in a single figure.
"""

import matplotlib.pyplot as plt
import os
import re
from OpenQuakeUHS.core.spectrum_parser import UHSSpectrum

def plot_uhs_sets(mean_files, quantile_files, rlz_files, poe=0.687, title=None):
    """
    Plots all UHS spectra grouped by type:
    - Realizations (rlz): thin gray lines, labeled once as 'all rlz'
    - Quantiles: dashed lines labeled as 'q16', 'q84', etc.
    - Mean: thick blue line labeled as 'Mean'

    Parameters:
    -----------
    mean_files : list of str
        Paths to CSV files containing 'mean' spectra.
    quantile_files : list of str
        Paths to quantile CSV files (e.g., 16th, 84th percentiles).
    rlz_files : list of str
        Paths to realization CSV files.
    poe : float
        Probability of exceedance to plot.
    title : str
        Optional title for the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ymax = 0
    rlz_plotted = False

    # Plot realizations
    for f in rlz_files:
        try:
            uhs = UHSSpectrum(f)
            T = uhs.mean.T()
            Sa = uhs.mean.Sa(poe)
            ymax = max(ymax, max(Sa))
            ax.plot(
                T, Sa,
                color='lightgray', linestyle='--', linewidth=0.8,
                label="all rlz" if not rlz_plotted else None
            )
            rlz_plotted = True
        except Exception as e:
            print(f"[rlz] Skipping {f}: {e}")

    # Plot quantiles
    for f in quantile_files:
        try:
            uhs = UHSSpectrum(f)
            T = uhs.mean.T()
            Sa = uhs.mean.Sa(poe)
            ymax = max(ymax, max(Sa))

            base = os.path.basename(f).lower()
            match = re.search(r"[-_](0\.\d+)", base)
            if match:
                quantile_val = float(match.group(1))
                label = f"q{int(round(quantile_val * 100))}"
            else:
                label = os.path.basename(f)

            ax.plot(T, Sa, linestyle='--', linewidth=1.2, label=label)
        except Exception as e:
            print(f"[quantile] Skipping {f}: {e}")

    # Plot mean
    lat, lon = None, None
    for f in mean_files:
        try:
            uhs = UHSSpectrum(f)
            T = uhs.mean.T()
            Sa = uhs.mean.Sa(poe)
            ymax = max(ymax, max(Sa))
            lat, lon = uhs.latitude, uhs.longitude
            ax.plot(T, Sa, color=[0, 0.4470, 0.7410], linewidth=2.0, label="Mean")
        except Exception as e:
            print(f"[mean] Skipping {f}: {e}")

    # Finalize plot
    ax.set_xlabel("Period [s]", fontweight="bold")
    ax.set_ylabel("Spectral Acceleration [g]", fontweight="bold")

    if title:
        ax.set_title(title, fontweight="bold")
    elif lat is not None and lon is not None:
        ax.set_title(f"UHS at ({lat:.3f}, {lon:.3f}) - PoE = {poe}", fontweight="bold")
    else:
        ax.set_title(f"UHS (PoE = {poe})", fontweight="bold")

    ax.set_xlim(0, 5)
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
