"""
File: disaggregation_hdf5.py
Author: Ing. Patricio Palacios Msc.
Date: April 2, 2026

Description:
------------
Defines Disaggregation_HDF5, an object-oriented interface to OpenQuake
diss.hdf5 outputs. Mirrors the existing Disaggregation CSV-based class.
"""

import json
import os

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


class Disaggregation_HDF5:
    """
    Object-oriented interface to an OpenQuake diss.hdf5 file.

    Parameters:
    -----------
    hdf5_path : str
        Path to the diss.hdf5 file.
    shp_path  : str or None
        Path to a shapefile to overlay on the Lon/Lat subplot.
    """

    def __init__(self, hdf5_path, shp_path=None):
        self.hdf5_path = hdf5_path
        self.shp_path  = shp_path
        self._load()

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _load(self):
        tag = "[Disaggregation_HDF5]"
        print(f"{tag} Loading: {os.path.basename(self.hdf5_path)}")

        with h5py.File(self.hdf5_path, "r") as f:

            # oqparam
            raw = f["oqparam"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            elif hasattr(raw, "item"):
                raw = raw.item()
            oqp = json.loads(raw)
            self.description = oqp.get("description", "")

            # site
            self.longitude = float(f["sitecol/lon"][0])
            self.latitude  = float(f["sitecol/lat"][0])

            # bins (edges)
            self._mag_bins  = f["disagg-bins/Mag"][:]
            self._dist_bins = f["disagg-bins/Dist"][:]
            self._eps_bins  = f["disagg-bins/Eps"][:]
            self._lon_bins  = f["disagg-bins/Lon"][0, :]
            self._lat_bins  = f["disagg-bins/Lat"][0, :]
            self.trts       = [t.decode() for t in f["disagg-bins/TRT"][:]]

            # available imts and poes
            meta            = json.loads(f["hmaps-stats"].attrs["json"])
            self.poes_avail = meta.get("poe", [])
            self.imts_avail = meta.get("imt", [])

            # store all disagg-stats matrices keyed by name
            self._matrices = {}
            for key in f["disagg-stats"]:
                self._matrices[key] = f[f"disagg-stats/{key}"][:]

        # bin centres
        self._mag_c  = 0.5 * (self._mag_bins[:-1]  + self._mag_bins[1:])
        self._dist_c = 0.5 * (self._dist_bins[:-1] + self._dist_bins[1:])
        self._eps_c  = 0.5 * (self._eps_bins[:-1]  + self._eps_bins[1:])
        self._lon_c  = 0.5 * (self._lon_bins[:-1]  + self._lon_bins[1:])
        self._lat_c  = 0.5 * (self._lat_bins[:-1]  + self._lat_bins[1:])

        tag = "[Disaggregation_HDF5]"
        print(f"{tag} Description      : {self.description}")
        print(f"{tag} Site             : lat={self.latitude:.4f}, lon={self.longitude:.4f}")
        print(f"{tag} IMTs available   : {', '.join(self.imts_avail)}")
        print(f"{tag} PoEs available   : {', '.join(str(p) for p in self.poes_avail)}")
        print(f"{tag} TRTs             : {', '.join(self.trts)}")
        print(f"{tag} Mag bins         : {self._mag_bins[0]:.1f} - {self._mag_bins[-1]:.1f} "
              f"({len(self._mag_c)} intervals)")
        print(f"{tag} Dist bins        : {self._dist_bins[0]:.0f} - {self._dist_bins[-1]:.0f} km "
              f"({len(self._dist_c)} intervals)")
        print(f"{tag} Eps bins         : {self._eps_bins[0]:.0f} to {self._eps_bins[-1]:.0f} "
              f"({len(self._eps_c)} intervals)")
        print(f"{tag} Lon/Lat grid     : {len(self._lon_c)}x{len(self._lat_c)}")
        shp_label = os.path.basename(self.shp_path) if self.shp_path else "None"
        print(f"{tag} Shapefile        : {shp_label}")
        print(f"{tag} Ready.")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _validate(self, target_poe, target_imt):
        if target_poe not in self.poes_avail:
            raise ValueError(
                f"PoE {target_poe} not available. Options: {self.poes_avail}"
            )
        if target_imt not in self.imts_avail:
            raise ValueError(
                f"IMT '{target_imt}' not available. Options: {self.imts_avail}"
            )

    def _get_indices(self, target_poe, target_imt):
        return self.poes_avail.index(target_poe), self.imts_avail.index(target_imt)

    def _build_data_mde(self, target_poe, target_imt):
        """Build Mag_Dist_Eps DataFrame."""
        poe_idx, imt_idx = self._get_indices(target_poe, target_imt)
        mde = self._matrices["Mag_Dist_Eps"][0, :, :, :, imt_idx, poe_idx, 0]
        rows = []
        for i, m in enumerate(self._mag_c):
            for j, d in enumerate(self._dist_c):
                for k, e in enumerate(self._eps_c):
                    rows.append({
                        "mag": float(m), "dist": float(d), "eps": float(e),
                        "mean": float(mde[i, j, k]),
                        "poe": target_poe, "imt": target_imt,
                    })
        return pd.DataFrame(rows)

    def _build_data_trt(self, target_poe, target_imt):
        """Build TRT_Mag_Dist DataFrame."""
        poe_idx, imt_idx = self._get_indices(target_poe, target_imt)
        tmd = self._matrices["TRT_Mag_Dist"][0, :, :, :, imt_idx, poe_idx, 0]
        rows = []
        for ti, trt in enumerate(self.trts):
            for i, m in enumerate(self._mag_c):
                for j, d in enumerate(self._dist_c):
                    rows.append({
                        "trt": trt, "mag": float(m), "dist": float(d),
                        "mean": float(tmd[ti, i, j]),
                        "poe": target_poe, "imt": target_imt,
                    })
        return pd.DataFrame(rows)

    def _build_data_lonlat(self, target_poe, target_imt):
        """Build TRT_Lon_Lat DataFrame."""
        poe_idx, imt_idx = self._get_indices(target_poe, target_imt)
        tll = self._matrices["TRT_Lon_Lat"][0, :, :, :, imt_idx, poe_idx, 0]
        rows = []
        for ti, trt in enumerate(self.trts):
            for i, lo in enumerate(self._lon_c):
                for j, la in enumerate(self._lat_c):
                    rows.append({
                        "trt": trt, "lon": float(lo), "lat": float(la),
                        "mean": float(tll[ti, i, j]),
                        "poe": target_poe, "imt": target_imt,
                    })
        return pd.DataFrame(rows)

    def _disagg_mod_mean(self, data):
        """Compute modal and mean Mag/Dist and return stacked array for bar3d."""
        data01    = data.copy()
        hz_key    = [c for c in data01.columns if c.startswith("mean") or c.startswith("rlz")][0]
        data01["hz_cont"] = data01[hz_key] / data01[hz_key].sum()

        reduced   = data01.groupby(["mag", "dist"])["hz_cont"].sum().reset_index()
        modal_row = reduced.sort_values("hz_cont", ascending=False).iloc[0]
        mod_mag, mod_dist = modal_row["mag"], modal_row["dist"]
        mean_mag  = np.sum(reduced["mag"]  * reduced["hz_cont"])
        mean_dist = np.sum(reduced["dist"] * reduced["hz_cont"])

        d01       = data01.pivot_table(
            values="hz_cont", index=["mag", "dist"],
            columns="eps", aggfunc=np.sum, fill_value=0
        ).reset_index()
        eps_vals  = list(d01.columns[2:])
        cmap      = cm.get_cmap("jet", len(eps_vals))
        clrs      = [cmap(i) for i in range(len(eps_vals))]

        d01.rename(columns=dict(zip(eps_vals,
                   [f"ep{i}" for i in range(len(eps_vals))])), inplace=True)
        for j in range(len(eps_vals)):
            d01[f"c{j}"] = [clrs[j]] * len(d01)
        d01["ez0"] = np.zeros(len(d01))
        for j in range(1, len(eps_vals)):
            d01[f"ez{j}"] = d01[f"ez{j-1}"] + d01[f"ep{j-1}"]

        dat = np.vstack([
            d01[["dist", "mag", f"ep{i}", f"ez{i}", f"c{i}"]].values
            for i in range(len(eps_vals))
        ])
        return dat, mod_mag, mod_dist, mean_mag, mean_dist, clrs, eps_vals

    def _disagg_trt(self, data_TRT):
        data_TRT = data_TRT.copy()
        data_TRT["hz_cont_TRT"] = data_TRT["mean"] / data_TRT["mean"].sum()
        trt_unique = data_TRT["trt"].unique()
        tab_colors = ["tab:blue", "tab:orange", "tab:green",
                      "tab:red", "tab:purple", "tab:brown"]
        trt_colors = {trt: tab_colors[i % len(tab_colors)]
                      for i, trt in enumerate(trt_unique)}
        data_TRT["color_TRT"] = data_TRT["trt"].map(trt_colors)
        return data_TRT, trt_colors, trt_unique

    def _disagg_lonlat(self, data_lon_lat):
        data_lon_lat = data_lon_lat.copy()
        data_lon_lat["hz_cont_lon_lat"] = (
            data_lon_lat["mean"] / data_lon_lat["mean"].sum()
        )
        trt_unique = data_lon_lat["trt"].unique()
        tab_colors = ["tab:blue", "tab:orange", "tab:green",
                      "tab:red", "tab:purple", "tab:brown"]
        trt_colors = {trt: tab_colors[i % len(tab_colors)]
                      for i, trt in enumerate(trt_unique)}
        data_lon_lat["color_TRT"] = data_lon_lat["trt"].map(trt_colors)
        return data_lon_lat

    # -------------------------------------------------------------------------
    # Public method
    # -------------------------------------------------------------------------
    def plot(self, 
                target_poe,
                target_imt, 
                PRY_name="PRY", 
                save_path=None):
        """
        Plot the three disaggregation subplots for the given PoE(s) and IMT.

        Parameters:
        -----------
        target_poe : float or list of float
        target_imt : str
        PRY_name   : str
        save_path  : str or None  — base path without extension
        """
        poes = target_poe if isinstance(target_poe, list) else [target_poe]
        for p in poes:
            self._plot_single(p, target_imt, PRY_name, save_path)


    def _plot_single(self, target_poe, target_imt, PRY_name, save_path):
        tag = "[Disaggregation_HDF5]"
        self._validate(target_poe, target_imt)
        print(f"{tag} plot | target_poe={target_poe} | target_imt={target_imt}")

        data        = self._build_data_mde(target_poe, target_imt)
        data_TRT    = self._build_data_trt(target_poe, target_imt)
        data_lonlat = self._build_data_lonlat(target_poe, target_imt)

        dat, mod_mag, mod_dist, mean_mag, mean_dist, clrs, eps_vals = \
            self._disagg_mod_mean(data)
        data_TRT, trt_colors, trt_unique = self._disagg_trt(data_TRT)
        data_lonlat = self._disagg_lonlat(data_lonlat)

        print(f"{tag} Modal scenario   : Mw={mod_mag:.2f}, R={mod_dist:.0f} km")
        print(f"{tag} Mean scenario    : Mw={mean_mag:.2f}, R={mean_dist:.0f} km")

        fig = plt.figure(figsize=(18, 6))

        # subplot 1: Mag-Dist-Eps
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.set_ylabel("Mw", fontweight="bold")
        ax1.set_xlabel("Distance (km)", fontweight="bold")
        ax1.set_zlabel("Hazard Contribution (%)", fontweight="bold")
        ax1.bar3d(
            x=dat[:, 0].astype(float), y=dat[:, 1].astype(float),
            z=dat[:, 3].astype(float) * 100,
            dx=15.0, dy=0.15, dz=dat[:, 2].astype(float) * 100,
            shade=True, color=dat[:, 4], zsort="max", alpha=0.5,
        )
        ax1.set_title(
            f"Disaggregation by Epsilon\n"
            f"Mod: Mw={mod_mag:.2f}, R={mod_dist:.0f} km | "
            f"Mean: Mw={mean_mag:.2f}, R={mean_dist:.0f} km",
            fontweight="bold",
        )

        # subplot 2: TRT-Mag-Dist
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.set_ylabel("Mw", fontweight="bold")
        ax2.set_xlabel("Distance (km)", fontweight="bold")
        ax2.set_zlabel("Hazard Contribution (%)", fontweight="bold")
        ax2.bar3d(
            x=data_TRT["dist"].astype(float), y=data_TRT["mag"].astype(float),
            z=0.0, dx=15.0, dy=0.15,
            dz=data_TRT["hz_cont_TRT"] * 100,
            color=data_TRT["color_TRT"], shade=True, zsort="max", alpha=0.8,
        )
        ax2.set_title(
            f"Disaggregation by TRT\n(poe={target_poe}, imt={target_imt})",
            fontweight="bold",
        )

        # subplot 3: Lon-Lat
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        dz = data_lonlat["hz_cont_lon_lat"] * 100
        ax3.bar3d(
            x=data_lonlat["lon"], y=data_lonlat["lat"],
            z=np.zeros_like(dz), dx=0.2, dy=0.2, dz=dz,
            color=data_lonlat["color_TRT"], shade=True, alpha=0.8,
        )

        if self.shp_path:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(self.shp_path).to_crs(epsg=4326)
                for geom in gdf.geometry:
                    if geom.geom_type == "Polygon":
                        x, y = geom.exterior.xy
                        ax3.plot(x, y, zs=0, zdir="z", color="black", linewidth=1)
                    elif geom.geom_type == "MultiPolygon":
                        for poly in geom.geoms:
                            x, y = poly.exterior.xy
                            ax3.plot(x, y, zs=0, zdir="z", color="black", linewidth=1)
            except Exception as e:
                print(f"{tag} Shapefile skipped: {e}")

        ax3.set_xlabel("Longitude", fontweight="bold")
        ax3.set_ylabel("Latitude", fontweight="bold")
        ax3.set_zlabel("Hazard Contribution (%)", fontweight="bold")
        ax3.set_title("Hazard Contribution by Location", fontweight="bold")

        handles_eps = [mpatches.Patch(color=clrs[i], label=str(eps_vals[i]))
                       for i in range(len(eps_vals))]
        handles_trt = [mpatches.Patch(color=trt_colors[trt], label=trt)
                       for trt in trt_unique]

        leg1 = fig.legend(handles=handles_eps, title="Epsilon",
                          loc="lower center", bbox_to_anchor=(0.25, -0.05),
                          ncol=int(len(eps_vals) / 2), frameon=False)
        leg1.get_title().set_fontweight("bold")
        leg2 = fig.legend(handles=handles_trt, title="Tectonic Region Type",
                          loc="lower center", bbox_to_anchor=(0.63, -0.01),
                          ncol=3, frameon=False)
        leg2.get_title().set_fontweight("bold")

        fig.text(0.99, -0.01,
                 f"PRY: {PRY_name}\n© 2025 - Patricio Palacios B.",
                 ha="right", va="top", fontsize=9,
                 color="gray", style="italic", multialignment="right")

        if save_path:
            poe_str        = f"{target_poe:.2f}".replace(".", "")
            save_path_full = f"{save_path}_POE_{poe_str}"
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(f"{save_path_full}_disaggregation.svg", format="svg", bbox_inches="tight")
            fig.savefig(f"{save_path_full}_disaggregation.pdf", format="pdf", bbox_inches="tight")
            print(f"{tag} Figures saved: {os.path.basename(save_path_full)}_disaggregation.svg / .pdf")

        plt.show()

        plt.close("all")