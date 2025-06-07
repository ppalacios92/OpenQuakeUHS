import matplotlib.pyplot as plt
import os
import re
from OpenQuakeUHS.core.spectrum_parser import UHSSpectrum

def plot_uhs_sets(mean_files, quantile_files, rlz_files, poe=0.687, title=None):
    """
    Plots all UHS spectra grouped by type with stylized formatting:
    - Realizations: gray dashed lines labeled once
    - Quantiles: dashed with unique markers and value in legend
    - Mean: blue solid line with circle markers and PGA value
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ymax = 0
    rlz_plotted = False
    lat, lon = None, None

    # --- Realizations ---
    for f in rlz_files:
        try:
            uhs = UHSSpectrum(f)
            T = uhs.mean.T()
            Sa = uhs.mean.Sa(poe)
            ymax = max(ymax, max(Sa))
            ax.plot(
                T, Sa,
                color='lightgray', linestyle='--', linewidth=0.8,
                label="All realizations" if not rlz_plotted else None
            )
            rlz_plotted = True
        except Exception as e:
            print(f"[rlz] Skipping {f}: {e}")

    # --- Quantiles ---
    markers = ['^', '<', '>']
    for i, f in enumerate(quantile_files):
        try:
            uhs = UHSSpectrum(f)
            T = uhs.mean.T()
            Sa = uhs.mean.Sa(poe)
            ymax = max(ymax, max(Sa))

            base = os.path.basename(f).lower()
            match = re.search(r"[-_](0\.\d+)", base)
            if match:
                quantile_val = float(match.group(1))
                label = f"Quantile-{quantile_val:.2f} / PGA={Sa[0]:.3f}g"
            else:
                label = f"Quantile / PGA={Sa[0]:.3f}g"

            ax.plot(
                T, Sa,
                linestyle='--',
                # marker=markers[i % len(markers)],
                linewidth=1.2,
                label=label
            )
        except Exception as e:
            print(f"[quantile] Skipping {f}: {e}")

    # --- Mean ---
    for f in mean_files:
        try:
            uhs = UHSSpectrum(f)
            T = uhs.mean.T()
            Sa = uhs.mean.Sa(poe)
            ymax = max(ymax, max(Sa))
            lat, lon = uhs.latitude, uhs.longitude
            label = f"Mean / PGA={Sa[0]:.3f}g"
            ax.plot(
                T, Sa,
                color=[0, 0.4470, 0.7410],
                linewidth=2.0,
                # marker='o',
                label=label
            )
        except Exception as e:
            print(f"[mean] Skipping {f}: {e}")

    # --- Formatting ---
    ax.set_xlabel("Period [s]", fontweight="bold")
    ax.set_ylabel("Sa [g]", fontweight="bold")

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
