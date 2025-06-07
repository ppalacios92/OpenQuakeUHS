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


def interpolate_sa_at_reference(y_values, sa_values, ref):
    """
    Interpola el valor de Sa para una referencia `ref` en PoE o 1/Tr,
    buscando los dos puntos más cercanos que lo encierren.

    Parámetros:
    -----------
    y_values : array-like
        Valores de PoE o 1/Tr (eje Y).
    sa_values : array-like
        Valores de aceleración espectral (eje X).
    ref : float
        Valor de referencia a interpolar.

    Retorna:
    --------
    float : Valor de Sa interpolado o np.nan si no hay cruce.
    """
    y_values = np.asarray(y_values)
    sa_values = np.asarray(sa_values)

    for i in range(len(y_values) - 1):
        y1, y2 = y_values[i], y_values[i + 1]
        x1, x2 = sa_values[i], sa_values[i + 1]

        # Comprobar que hay cruce (independiente del orden)
        if (y1 - ref) * (y2 - ref) <= 0 and y1 != y2:
            # Interpolación lineal entre (x1,y1) y (x2,y2)
            return x1 + (ref - y1) * (x2 - x1) / (y2 - y1)

    return np.nan  # si no se encuentra cruce



def plot_mean_and_rlz_hazard_curves(mean_files, rlz_files=None, periods=None, title=None, reference_value=None):
    investigation_time = 50
    lat, lon = None, None
    rlz_labeled = False

    # ----------------- FIGURA 1: Sa vs PoE -----------------
    fig1, ax1 = plt.subplots(figsize=(6, 4))

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

    print("\n--- Interpolación Sa vs PoE ---")
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

            ax1.plot(sa, poes, '-o' if is_pga else '-', linewidth=1.8 if is_pga else 1.5,
                     color='red' if is_pga else None, label=label)

            if reference_value:
                sa_ref = interpolate_sa_at_reference(poes, sa, reference_value)
                print(f"{label}: Sa interpolado para PoE={reference_value:.3f} → {sa_ref:.4f} g")

        except Exception as e:
            print(f"[mean - PoE] Skipping {f}: {e}")

    if reference_value:
        ax1.axhline(reference_value, color='black', linestyle='--', linewidth=1.2)

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
    rlz_labeled = False

    # Convertir PoE de entrada a 1/Tr
    ref_inv_Tr = None
    if reference_value:
        ref_inv_Tr = calculate_inv_Tr_from_poes([reference_value])[0]

    print("\n--- Interpolación Sa vs 1/Tr ---")
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

            ax2.plot(sa, inv_Tr, '-o' if is_pga else '-', linewidth=1.8 if is_pga else 1.5,
                     color='red' if is_pga else None, label=label)

            if reference_value:
                sa_ref = interpolate_sa_at_reference(inv_Tr, sa, ref_inv_Tr)
                print(f"{label}: Sa interpolado para 1/Tr={ref_inv_Tr:.5f} → {sa_ref:.4f} g")

        except Exception as e:
            print(f"[mean - inv_Tr] Skipping {f}: {e}")

    if reference_value:
        ax2.axhline(ref_inv_Tr, color='black', linestyle='--', linewidth=1.2)

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
