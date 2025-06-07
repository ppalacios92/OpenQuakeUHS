import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import numpy as np


def plot_mean_and_rlz_hazard_curves(mean_files, rlz_files=None, periods=None, title=None):
    """
    Plots hazard curves from mean files and overlays all realizations in gray.

    Parameters:
    -----------
    mean_files : list of str
        List of mean hazard curve CSV files.
    rlz_files : list of str or None
        List of realization hazard curve CSV files.
    periods : list of float or None
        Periods to filter (e.g., [0.1, 0.2]). If None, plot all.
    title : str
        Optional plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    lat, lon = None, None
    investigation_time = 50

    # --- GRAFICAR REALIZACIONES ---
    rlz_labeled = False
    if rlz_files:
        for f in rlz_files:
            try:
                filename = os.path.basename(f)

                # Detectar tipo de periodo
                if "PGA" in filename:
                    is_pga = True
                else:
                    match = re.search(r"SA\(([\d.]+)\)", filename)
                    if match:
                        T = float(match.group(1))
                        is_pga = False
                    else:
                        continue

                if periods is not None:
                    if is_pga and 0.01 not in periods:
                        continue
                    elif not is_pga and T not in periods:
                        continue

                df = pd.read_csv(f, header=None)
                sa_raw = df.iloc[1, 3:].astype(str)
                sa = sa_raw.str.replace("poe-", "", regex=False).astype(float)
                poes = df.iloc[2, 3:].astype(float)

                # Graficar realizaciones
                label = "All rlz" if not rlz_labeled else None
                ax.plot(sa, poes, color="lightgray", linewidth=0.8, label=label)
                rlz_labeled = True

            except Exception as e:
                print(f"[rlz] Skipping {f}: {e}")

    # --- GRAFICAR CURVAS MEAN ---
    for f in mean_files:
        try:
            filename = os.path.basename(f)

            # Extraer periodo
            if "PGA" in filename:
                label = "PGA"
                is_pga = True
            else:
                match = re.search(r"SA\(([\d.]+)\)", filename)
                if match:
                    T = float(match.group(1))
                    label = f"SA({T:.1f})"
                    is_pga = False
                else:
                    continue

            if periods is not None:
                if is_pga and 0.01 not in periods:
                    continue
                elif not is_pga and T not in periods:
                    continue

            df = pd.read_csv(f, header=None)
            sa_raw = df.iloc[1, 3:].astype(str)
            sa = sa_raw.str.replace("poe-", "", regex=False).astype(float)
            poes = df.iloc[2, 3:].astype(float)

            lon = float(df.iloc[2, 0])
            lat = float(df.iloc[2, 1])

            if is_pga:
                ax.plot(sa, poes, '-o', color='red', linewidth=1.8, label=label)
            else:
                ax.plot(sa, poes, linewidth=1.5, label=label)

        except Exception as e:
            print(f"[mean] Skipping {f}: {e}")

    # --- CONFIGURACIÓN FINAL ---
    ax.set_xlabel("Spectral Acceleration [g]", fontweight="bold")
    ax.set_ylabel("Probability of Exceedance in 50y", fontweight="bold")
    ax.set_title(
        title or f"Mean Hazard Curves at ({lat:.3f}, {lon:.3f})" if lat and lon else "Mean Hazard Curves",
        fontweight="bold"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 2.0)
    ax.set_ylim(1e-4, 1.1)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
