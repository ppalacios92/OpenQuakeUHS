import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import numpy as np


def calculate_inv_Tr_from_poes(poes, N=50):
    poes = np.asarray(poes, dtype=float)
    inv_Tr = np.zeros_like(poes)
    for i, pi in enumerate(poes):
        if pi <= 0 or pi >= 1:
            inv_Tr[i] = np.nan
        else:
            inv_Tr[i] = 1 - (1 - pi) ** (1 / N)
    return inv_Tr


def plot_mean_and_rlz_hazard_curves(mean_files, rlz_files=None, periods=None, title=None):
    # ----------------- FIGURA 1: Sa vs PoE -----------------
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    lat, lon = None, None
    investigation_time = 50
    rlz_labeled = False

    if rlz_files:
        for f in rlz_files:
            try:
                filename = os.path.basename(f)
                is_pga = "PGA" in filename
                if not is_pga:
                    match = re.search(r"SA\(([\d.]+)\)", filename)
                    if not match:
                        continue
                    T = float(match.group(1))

                if periods is not None:
                    if is_pga and 0.01 not in periods:
                        continue
                    elif not is_pga and T not in periods:
                        continue

                df = pd.read_csv(f, header=None)
                sa = df.iloc[1, 3:].astype(str).str.replace("poe-", "", regex=False).astype(float)
                poes = df.iloc[2, 3:].astype(float)

                label = "All rlz" if not rlz_labeled else None
                ax1.plot(sa, poes, color="lightgray", linewidth=0.8, label=label)
                rlz_labeled = True

            except Exception as e:
                print(f"[rlz - PoE] Skipping {f}: {e}")

    for f in mean_files:
        try:
            filename = os.path.basename(f)
            is_pga = "PGA" in filename
            if not is_pga:
                match = re.search(r"SA\(([\d.]+)\)", filename)
                if not match:
                    continue
                T = float(match.group(1))

            if periods is not None:
                if is_pga and 0.01 not in periods:
                    continue
                elif not is_pga and T not in periods:
                    continue

            df = pd.read_csv(f, header=None)
            sa = df.iloc[1, 3:].astype(str).str.replace("poe-", "", regex=False).astype(float)
            poes = df.iloc[2, 3:].astype(float)

            lon = float(df.iloc[2, 0])
            lat = float(df.iloc[2, 1])
            label = "PGA" if is_pga else f"SA({T:.1f})"

            if is_pga:
                ax1.plot(sa, poes, '-o', color='red', linewidth=1.8, label=label)
            else:
                ax1.plot(sa, poes, linewidth=1.5, label=label)

        except Exception as e:
            print(f"[mean - PoE] Skipping {f}: {e}")

    ax1.set_yscale("log")
    ax1.set_xlim(0.01, 2.0)
    ax1.set_ylim(1e-4, 1.0)
    ax1.set_xlabel("Spectral Acceleration [g]", fontweight="bold")
    ax1.set_ylabel("PoE in 50y", fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.set_title(
        title or f"Mean Hazard Curves at ({lat:.3f}, {lon:.3f})" if lat and lon else "Mean Hazard Curves",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # ----------------- FIGURA 2: Sa vs 1/Tr -----------------
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    rlz_labeled = False  # reiniciar etiqueta

    if rlz_files:
        for f in rlz_files:
            try:
                filename = os.path.basename(f)
                is_pga = "PGA" in filename
                if not is_pga:
                    match = re.search(r"SA\(([\d.]+)\)", filename)
                    if not match:
                        continue
                    T = float(match.group(1))

                if periods is not None:
                    if is_pga and 0.01 not in periods:
                        continue
                    elif not is_pga and T not in periods:
                        continue

                df = pd.read_csv(f, header=None)
                sa = df.iloc[1, 3:].astype(str).str.replace("poe-", "", regex=False).astype(float)
                poes = df.iloc[2, 3:].astype(float)
                inv_Tr = calculate_inv_Tr_from_poes(poes, N=investigation_time)

                label = "All rlz" if not rlz_labeled else None
                ax2.plot(sa, inv_Tr, color="lightgray", linewidth=0.8, label=label)
                rlz_labeled = True

            except Exception as e:
                print(f"[rlz - inv_Tr] Skipping {f}: {e}")

    for f in mean_files:
        try:
            filename = os.path.basename(f)
            is_pga = "PGA" in filename
            if not is_pga:
                match = re.search(r"SA\(([\d.]+)\)", filename)
                if not match:
                    continue
                T = float(match.group(1))

            if periods is not None:
                if is_pga and 0.01 not in periods:
                    continue
                elif not is_pga and T not in periods:
                    continue

            df = pd.read_csv(f, header=None)
            sa = df.iloc[1, 3:].astype(str).str.replace("poe-", "", regex=False).astype(float)
            poes = df.iloc[2, 3:].astype(float)
            inv_Tr = calculate_inv_Tr_from_poes(poes, N=investigation_time)

            label = "PGA" if is_pga else f"SA({T:.1f})"

            if is_pga:
                ax2.plot(sa, inv_Tr, '-o', color='red', linewidth=1.8, label=label)
            else:
                ax2.plot(sa, inv_Tr, linewidth=1.5, label=label)

        except Exception as e:
            print(f"[mean - inv_Tr] Skipping {f}: {e}")

    ax2.set_yscale("log")
    ax2.set_xlim(0.01, 2.0)
    ax2.set_ylim(1e-5, 0.1)
    ax2.set_xlabel("Spectral Acceleration [g]", fontweight="bold")
    ax2.set_ylabel("Annual Rate Excedence", fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()
    ax2.set_title(
        title or f"Mean Hazard Curves at ({lat:.3f}, {lon:.3f})" if lat and lon else "Mean Hazard Curves",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
