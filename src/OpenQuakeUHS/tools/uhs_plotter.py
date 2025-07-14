import matplotlib.pyplot as plt
import os
import re
from OpenQuakeUHS.core.spectrum_parser import UHSSpectrum

def plot_uhs_sets(mean_files, quantile_files=None, rlz_files=None, poe=[0.687], title=None , save_path=None):
    """
    Plots UHS spectra for multiple PoEs in a single figure:
    - Realizations: dashed gray lines, labeled once per PoE
    - Quantiles: dashed lines with legend including quantile and PoE
    - Mean: solid line with legend showing PGA and PoE

    Includes:
    - One figure with linear X scale
    - One figure with log X scale
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig_log, ax_log = plt.subplots(figsize=(6, 4))
    lat, lon = None, None
    ymax = 0

    for p in poe:
        rlz_plotted = False

        # --- Realizaciones ---
        if rlz_files is not None:
            for f in rlz_files:
                try:
                    uhs = UHSSpectrum(f)
                    T = uhs.mean.T()
                    Sa = uhs.mean.Sa(p)
                    label = f"All realizations (PoE={p})" if not rlz_plotted else None
                    ax.plot(T, Sa, color="lightgray", linestyle="--", linewidth=0.8, label=label)
                    ax_log.plot(T, Sa, color="lightgray", linestyle="--", linewidth=0.8, label=label)
                    ymax = max(ymax, max(Sa))
                    rlz_plotted = True
                except Exception as e:
                    print(f"[rlz] Skipping {f}: {e}")

        # --- Cuantiles ---
        if quantile_files is not None:
            for f in quantile_files:
                try:
                    uhs = UHSSpectrum(f)
                    T = uhs.mean.T()
                    Sa = uhs.mean.Sa(p)

                    base = os.path.basename(f).lower()
                    match = re.search(r"[-_](0\.\d+)", base)
                    if match:
                        q = float(match.group(1))
                        label = f"Quantile-{q:.2f} (PoE={p}) / PGA={Sa[0]:.3f}g"
                    else:
                        label = f"Quantile (PoE={p}) / PGA={Sa[0]:.3f}g"

                    ax.plot(T, Sa, linestyle='--', linewidth=1.2, label=label)
                    ax_log.plot(T, Sa, linestyle='--', linewidth=1.2, label=label)
                    ymax = max(ymax, max(Sa))
                except Exception as e:
                    print(f"[quantile] Skipping {f}: {e}")

        # --- Medias ---
        for f in mean_files:
            try:
                uhs = UHSSpectrum(f)
                T = uhs.mean.T()
                Sa = uhs.mean.Sa(p)
                lat, lon = uhs.latitude, uhs.longitude
                label = f"Mean (PoE={p}) / PGA={Sa[0]:.3f}g"

                ax.plot(T, Sa, linewidth=2.0, label=label)
                ax_log.plot(T, Sa, linewidth=2.0, label=label)
                ymax = max(ymax, max(Sa))
            except Exception as e:
                print(f"[mean] Skipping {f}: {e}")

    # --- Configurar gráfico lineal ---
    ax.set_xlabel("Period [s]", fontweight="bold")
    ax.set_ylabel("Sa [g]", fontweight="bold")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
    ax.grid(True)

    if title:
        ax.set_title(title, fontweight="bold")
    elif lat is not None and lon is not None:
        ax.set_title(f"UHS at ({lat:.3f}, {lon:.3f})", fontweight="bold")
    else:
        ax.set_title("UHS spectra", fontweight="bold")

    # Leyenda fuera del gráfico lineal
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)

    # --- Configurar gráfico logarítmico ---
    ax_log.set_xlabel("Period [s] (log scale)", fontweight="bold")
    ax_log.set_ylabel("Sa [g]", fontweight="bold")
    ax_log.set_xscale("log")
    ax_log.set_xlim(0.01, 5)
    ax_log.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
    ax_log.grid(True, which="both", linestyle="--", alpha=0.5)

    if title:
        ax_log.set_title(f"{title} (log-X)", fontweight="bold")
    elif lat is not None and lon is not None:
        ax_log.set_title(f"UHS at ({lat:.3f}, {lon:.3f}) - log scale", fontweight="bold")
    else:
        ax_log.set_title("UHS spectra - log scale", fontweight="bold")

    # Leyenda fuera del gráfico log-X
    ax_log.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)

    fig.subplots_adjust(right=0.75)
    fig_log.subplots_adjust(right=0.75)

    # plt.show()
    # --- Guardar si se especifica save_path ---
    if save_path:
        fig.savefig(f"{save_path}_linear.svg", format="svg", bbox_inches="tight")
        fig.savefig(f"{save_path}_linear.pdf", format="pdf", bbox_inches="tight")
        fig_log.savefig(f"{save_path}_log.svg", format="svg", bbox_inches="tight")
        fig_log.savefig(f"{save_path}_log.pdf", format="pdf", bbox_inches="tight")
        print(f"Figures saved to {save_path}_linear.(svg/pdf) and {save_path}_log.(svg/pdf)")

    else:
        plt.show()