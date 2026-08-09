"""
File: psha_hdf5.py
Author: Ing. Patricio Palacios Msc.
Date: April 2, 2026

Description:
------------
Defines PSHA_HDF5, an object-oriented interface to OpenQuake psha.hdf5 outputs.
Provides UHS plotting, hazard curve plotting, UHS table generation, and site map.
All methods mirror the existing CSV-based pipeline signatures.
"""

import json
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from OpenQuakeUHS.tools.map_utils import plot_uhs_location_map
from OpenQuakeUHS.tools.hazard_plotter import (
    calculate_inv_Tr_from_poes,
    interpolate_sa_at_reference,
)


class PSHA_HDF5:
    """
    Object-oriented interface to an OpenQuake psha.hdf5 file.

    Works with both a single-site datastore and a multi-site one: the site of interest is
    chosen at load time and every array is sliced down to it, so all methods below operate on
    one site regardless of how many the file contains.

    Parameters:
    -----------
    hdf5_path : str
        Path to the psha.hdf5 file.
    site : tuple, int or None
        Which site to read.
        - None       : first site in the file (the historical behaviour).
        - (lon, lat) : the site closest to that coordinate.
        - int        : that site index directly.

        Selecting by coordinate is the safe option on a multi-site run, because OpenQuake
        REORDERS the sites internally: the position a site has in the job.ini `sites` list is
        not the position it has in the datastore. Never assume the input order survives.

    Examples:
    ---------
    >>> psha = PSHA_HDF5('psha.hdf5', site=(-78.4851, -0.1753))   # Quito
    >>> psha.plot_uhs_sets(poe=[0.5, 0.2, 0.1, 0.02], show_rlzs=True)
    """

    def __init__(self, hdf5_path, site=None):
        self.hdf5_path = hdf5_path
        self._load(site)

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_site(lons, lats, site):
        """Return (index, distance_in_degrees) of the requested site.

        The distance is reported so the caller can tell a hit from a near miss: asking for a
        city and silently getting the one 80 km away is the failure mode this guards against.
        """
        if site is None:
            return 0, 0.0
        if isinstance(site, (int, np.integer)):
            if not 0 <= site < len(lons):
                raise IndexError(f"site index {site} out of range, file has {len(lons)} sites")
            return int(site), 0.0

        lon, lat = site
        d = np.hypot(lons - lon, lats - lat)
        i = int(np.argmin(d))
        return i, float(d[i])

    def _load(self, site=None):
        tag = "[PSHA_HDF5]"
        print(f"{tag} Loading: {os.path.basename(self.hdf5_path)}")

        with h5py.File(self.hdf5_path, "r") as f:

            # oqparam
            raw = f["oqparam"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            elif hasattr(raw, "item"):
                raw = raw.item()
            oqp = json.loads(raw)

            self.description       = oqp.get("description", "")
            self.investigation_time = float(oqp.get("investigation_time", 50.0))
            self.imtls             = oqp["hazard_imtls"]   # {imt: [iml, ...]}

            # site: resolve which one to read, then keep only that one
            lons = f["sitecol/lon"][:]
            lats = f["sitecol/lat"][:]
            self.n_sites = len(lons)
            i, self.site_distance = self._resolve_site(lons, lats, site)
            self.site_index = i

            self.longitude = float(lons[i])
            self.latitude  = float(lats[i])
            self.depth     = float(f["sitecol/depth"][i])
            self.vs30      = float(f["sitecol/vs30"][i])

            # hmaps-stats meta
            ms_meta   = json.loads(f["hmaps-stats"].attrs["json"])
            self.imts  = ms_meta["imt"]
            self.stats = ms_meta["stat"]
            self.poes  = ms_meta["poe"]

            # hmaps-rlzs meta
            mr_meta      = json.loads(f["hmaps-rlzs"].attrs["json"])
            self.n_rlzs  = mr_meta["rlz_id"]

            # hcurves-stats meta
            hcs_meta = json.loads(f["hcurves-stats"].attrs["json"])

            # logic tree
            try:
                sm_data   = f["full_lt/sm_data"][:]
                self.n_sm = len(sm_data)
            except Exception:
                self.n_sm = f["full_lt"].attrs.get("num_samples", 0)
    
            trt_raw   = f["full_lt"].attrs.get("trts", [])
            self.trts = list(trt_raw) if not isinstance(trt_raw, str) else json.loads(trt_raw)

            # weights
            self.weights = f["weights"][:]

            # Arrays are sliced to the selected site with i:i+1, which KEEPS the leading axis
            # at length 1. Every method below therefore indexes [0, ...] exactly as it did when
            # the file could only hold one site, and none of them needed changing.
            self._hmaps_stats = f["hmaps-stats"][i:i + 1]   # (1, n_stats, n_imts, n_poes)
            self._hmaps_rlzs  = f["hmaps-rlzs"][i:i + 1]    # (1, n_rlzs,  n_imts, n_poes)

            self._hcurves_stats = f["hcurves-stats"][i:i + 1]
            self._hcurves_rlzs  = f["hcurves-rlzs"][i:i + 1]

        # derived
        self._stat_idx  = {s: i for i, s in enumerate(self.stats)}
        self._imt_idx   = {m: i for i, m in enumerate(self.imts)}
        self._poe_idx   = {p: i for i, p in enumerate(self.poes)}

        # print summary
        tag = "[PSHA_HDF5]"
        print(f"{tag} Description       : {self.description}")
        print(f"{tag} Site              : lat={self.latitude:.4f}, lon={self.longitude:.4f}, "
              f"vs30={self.vs30:.1f} m/s, depth={self.depth:.1f} km")
        if self.n_sites > 1:
            print(f"{tag} Site selection    : index {self.site_index} of {self.n_sites}"
                  + (f", {self.site_distance:.4f} deg from the requested coordinate"
                     if site is not None and not isinstance(site, (int, np.integer)) else ""))
            if site is None:
                print(f"{tag} WARNING: this file holds {self.n_sites} sites and none was "
                      f"requested, so the first one was taken. Pass site=(lon, lat) to choose.")
        print(f"{tag} Investigation time: {self.investigation_time:.0f} years")
        print(f"{tag} IMTs ({len(self.imts):2d})         : {', '.join(self.imts)}")
        print(f"{tag} PoEs ({len(self.poes):2d})          : {', '.join(str(p) for p in self.poes)}")
        print(f"{tag} IML levels/IMT   : {self._hcurves_stats.shape[3]}")
        print(f"{tag} Stats             : {', '.join(self.stats)}")
        print(f"{tag} Realizations      : {self.n_rlzs}")
        print(f"{tag} Logic tree paths  : {self.n_sm} source models x {len(self.trts)} TRTs")
        print(f"{tag} TRTs              : {', '.join(self.trts)}")
        print(f"{tag} Ready.")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _imt_to_period(self, imt_str):
        imt_str = imt_str.strip()
        if imt_str == "PGA":
            return 0.01
        m = re.match(r"SA\(([\d.]+)\)", imt_str)
        return float(m.group(1)) if m else float("nan")

    def _get_uhs_obj(self, stat=None, rlz_id=None):
        """Return a lightweight dict {poe: {period: Sa}} for given stat or rlz."""
        from collections import defaultdict
        data = defaultdict(dict)

        for poe_idx, poe in enumerate(self.poes):
            for imt_idx, imt_str in enumerate(self.imts):
                period = self._imt_to_period(imt_str)
                if np.isnan(period):
                    continue
                if rlz_id is not None:
                    sa = float(self._hmaps_rlzs[0, rlz_id, imt_idx, poe_idx])
                else:
                    si = self._stat_idx[stat]
                    sa = float(self._hmaps_stats[0, si, imt_idx, poe_idx])
                data[poe][period] = sa
        return data

    def _periods_from_data(self, data):
        all_p = [set(v.keys()) for v in data.values()]
        return sorted(set.intersection(*all_p))

    def _sa_from_data(self, data, poe, periods):
        return [data[poe][T] for T in periods]

    def _quantile_label(self, stat):
        m = re.search(r"([\d.]+)$", stat)
        if m:
            return f"q{int(round(float(m.group(1)) * 100))}"
        return stat

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def plot_uhs_sets(self, 
                        poe, PRY_name="PRY", 
                        title=None,
                        show_rlzs=False, quantiles="all",
                        save_path=None):
        """
        Plot UHS sets (mean + quantiles + optional rlzs) for the given PoEs.

        Parameters:
        -----------
        poe       : list of float
            Probabilities of exceedance to plot (e.g. [0.5, 0.2, 0.1, 0.02]).
        PRY_name  : str
            Project name shown in the figure watermark.
        title     : str or None
            Custom figure title. Defaults to site coordinates.
        show_rlzs : bool
            If True, plot all individual realizations as thin gray dashed lines.
        quantiles : str, list of str, or None
            Controls which quantile curves to plot.
            - 'all'            : plot all available quantiles (default)
            - ['q16', 'q84']   : plot only the specified quantiles
            - None             : do not plot any quantiles
        save_path : str or None
            Base path without extension. If provided, saves four files:
            {save_path}_linear.svg/.pdf and {save_path}_log.svg/.pdf
        """

        tag = "[PSHA_HDF5]"
        q_stats = [s for s in self.stats if s.startswith("quantile")]
        print(f"{tag} plot_uhs_sets | PoEs: {poe} | stats: mean + "
              f"{len(q_stats)} quantiles + "
              f"{'%d rlzs' % self.n_rlzs if show_rlzs else 'no rlzs'}")

        fig,     ax     = plt.subplots(figsize=(6, 4))
        fig_log, ax_log = plt.subplots(figsize=(6, 4))
        ymax = 0

        for fig_obj in (fig, fig_log):
            fig_obj.text(0.99, -0.01,
                         f"PRY: {PRY_name}\n© 2025 - Patricio Palacios B.",
                         ha="right", va="top", fontsize=9,
                         color="gray", style="italic", multialignment="right")

        # color palette — one color per PoE
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        poe_colors = {p: prop_cycle[i % len(prop_cycle)] for i, p in enumerate(poe)}

        # realizations
        rlz_plotted = False
        if show_rlzs:
            for rlz_id in range(self.n_rlzs):
                d = self._get_uhs_obj(rlz_id=rlz_id)
                T = self._periods_from_data(d)
                for p in poe:
                    Sa = self._sa_from_data(d, p, T)
                    lbl = f"All realizations" if not rlz_plotted else None
                    ax.plot(T, Sa, color="lightgray", linestyle="--",
                            linewidth=0.8, label=lbl, zorder=1)
                    ax_log.plot(T, Sa, color="lightgray", linestyle="--",
                                linewidth=0.8, label=lbl, zorder=1)
                    ymax = max(ymax, max(Sa))
                    rlz_plotted = True


        # quantiles
        if quantiles is not None:
            for stat in q_stats:
                lbl_key = self._quantile_label(stat)
                if quantiles != "all" and lbl_key not in quantiles:
                    continue
                d = self._get_uhs_obj(stat=stat)
                T = self._periods_from_data(d)
                for p in poe:
                    Sa = self._sa_from_data(d, p, T)
                    lbl = f"{lbl_key} (PoE={p}) / PGA={Sa[0]:.3f}g"
                    ax.plot(T, Sa, linestyle="--", linewidth=1.2, label=lbl,
                            color=poe_colors[p], alpha=0.5)
                    ax_log.plot(T, Sa, linestyle="--", linewidth=1.2, label=lbl,
                                color=poe_colors[p], alpha=0.5)
                    ymax = max(ymax, max(Sa))

        # mean
        d_mean = self._get_uhs_obj(stat="mean")
        T_mean = self._periods_from_data(d_mean)
        for p in poe:
            Sa = self._sa_from_data(d_mean, p, T_mean)
            lbl = f"Mean (PoE={p}) / PGA={Sa[0]:.3f}g"
            ax.plot(T_mean, Sa, linewidth=2.0, label=lbl, color=poe_colors[p])
            ax_log.plot(T_mean, Sa, linewidth=2.0, label=lbl, color=poe_colors[p])
            ymax = max(ymax, max(Sa))

        site_title = title or f"UHS at ({self.latitude:.3f}, {self.longitude:.3f}) | vs30={self.vs30:.0f} m/s"

        for ax_obj, log in ((ax, False), (ax_log, True)):
            ax_obj.set_xlabel("Period [s]" + (" (log scale)" if log else ""),
                              fontweight="bold")
            ax_obj.set_ylabel("Sa [g]", fontweight="bold")
            ax_obj.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
            ax_obj.grid(True, which="both" if log else "major",
                        linestyle="--" if log else "-", alpha=0.5)
            ax_obj.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
            ax_obj.set_title(site_title + (" (log-X)" if log else ""),
                             fontweight="bold")
            if log:
                ax_obj.set_xscale("log")
                ax_obj.set_xlim(0.01, 5)
            else:
                ax_obj.set_xlim(0, 5)

        fig.subplots_adjust(right=0.75)
        fig_log.subplots_adjust(right=0.75)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(f"{save_path}_linear.svg",     format="svg", bbox_inches="tight")
            fig.savefig(f"{save_path}_linear.pdf",     format="pdf", bbox_inches="tight")
            fig_log.savefig(f"{save_path}_log.svg",    format="svg", bbox_inches="tight")
            fig_log.savefig(f"{save_path}_log.pdf",    format="pdf", bbox_inches="tight")
            print(f"{tag} Figures saved: {os.path.basename(save_path)}_linear.svg / "
                  f"_linear.pdf / _log.svg / _log.pdf")
        plt.show()


        # build return dataframe
        records = []
        d_mean = self._get_uhs_obj(stat="mean")
        T_mean = self._periods_from_data(d_mean)

        for p in poe:
            for T_val in T_mean:
                row = {"poe": p, "T": T_val,
                       "mean": self._get_uhs_obj(stat="mean")[p][T_val]}
                if quantiles is not None:
                    for stat in q_stats:
                        lbl_key = self._quantile_label(stat)
                        if quantiles != "all" and lbl_key not in quantiles:
                            continue
                        row[lbl_key] = self._get_uhs_obj(stat=stat)[p][T_val]
                records.append(row)

        df_out = pd.DataFrame(records)

        plt.show()
        plt.close("all")

        return df_out


    def generate_uhs_table(self, poe, PRY_name="PRY", save_path=None):
        """
        Generate UHS table(s) for one or multiple PoEs.
        Columns: T, mean, q16, q50, q84.

        Parameters:
        -----------
        poe       : float or list of float
        PRY_name  : str
        save_path : str or None

        Returns:
        --------
        pandas.DataFrame if poe is float, dict {poe: DataFrame} if poe is list
        """
        tag     = "[PSHA_HDF5]"
        q_stats = [s for s in self.stats if s.startswith("quantile")]
        poes    = poe if isinstance(poe, list) else [poe]

        print(f"{tag} generate_uhs_table | PoEs: {poes} | "
              f"columns: T, mean, {', '.join(self._quantile_label(s) for s in q_stats)}")

        d_mean  = self._get_uhs_obj(stat="mean")
        T       = self._periods_from_data(d_mean)
        results = {}

        for p in poes:
            data = {"T": T, "mean": self._sa_from_data(d_mean, p, T)}
            for stat in q_stats:
                d        = self._get_uhs_obj(stat=stat)
                data[self._quantile_label(stat)] = self._sa_from_data(d, p, T)

            df = pd.DataFrame(data)

            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
                poe_str  = f"{p:.2f}".replace(".", "")
                csv_name = f"{save_path}_UHS_Table_POE_{poe_str}_{PRY_name}.csv"
                df.to_csv(csv_name, index=False)
                print(f"{tag} Table saved: {os.path.basename(csv_name)}")

            results[p] = df

        return results if isinstance(poe, list) else results[poes[0]]

        
    def generate_uhs_peer(self, poe, stat="mean", PRY_name="PRY", save_path=None):
        """
        Export UHS in PEER ground motion database format.

        Parameters:
        -----------
        poe       : float or list of float
        stat      : str or list of str  — 'mean', 'q16', 'q50', 'q84'
        PRY_name  : str
        save_path : str or None
        """
        tag     = "[PSHA_HDF5]"
        poes    = poe  if isinstance(poe,  list) else [poe]
        stats_r = stat if isinstance(stat, list) else [stat]

        # map short label -> internal stat string
        label_to_stat = {"mean": "mean"}
        label_to_stat.update({
            self._quantile_label(s): s
            for s in self.stats if s.startswith("quantile")
        })

        print(f"{tag} generate_uhs_peer | PoEs: {poes} | stats: {stats_r}")

        for p in poes:
            for lbl in stats_r:
                if lbl not in label_to_stat:
                    print(f"{tag} WARNING: stat '{lbl}' not found, skipping")
                    continue

                d  = self._get_uhs_obj(stat=label_to_stat[lbl])
                T  = self._periods_from_data(d)
                Sa = self._sa_from_data(d, p, T)

                poe_str = f"{p:.2f}".replace(".", "")
                title   = f"UHS_{lbl}_POE_{poe_str}_{PRY_name}"

                rows = [
                    [title, ""],
                    ["", ""],
                    ["T (s)", "Sa (g)"],
                ] + [[t, s] for t, s in zip(T, Sa)]

                df = pd.DataFrame(rows)

                if save_path:
                    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
                    csv_name = f"{save_path}_PEER_{title}.csv"
                    df.to_csv(csv_name, index=False, header=False)
                    print(f"{tag} PEER file saved: {os.path.basename(csv_name)}")


    def plot_hazard_curves(self, 
                            periods, reference_value=None,
                            title=None, 
                            PRY_name="PRY", save_path=None):
        """
        Plot mean hazard curves (Sa vs PoE and Sa vs annual exceedance rate)
        for the requested periods.

        Parameters:
        -----------
        periods         : list of float  (e.g. [0.01, 0.2, 0.5, 1.0])
        reference_value : list of float or None  (PoE lines to draw)
        title           : str or None
        PRY_name        : str
        save_path       : str or None
        """
        tag = "[PSHA_HDF5]"
        print(f"{tag} plot_hazard_curves | periods: {periods} | "
              f"refs: {reference_value}")

        N          = self.investigation_time
        stat_idx   = self._stat_idx["mean"]
        ref_inv_Tr = calculate_inv_Tr_from_poes(reference_value, N=N) \
                     if reference_value else []

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for fig_obj in (fig1, fig2):
            fig_obj.text(0.99, -0.01,
                         f"PRY: {PRY_name}\n© 2025 - Patricio Palacios B.",
                         ha="right", va="top", fontsize=9,
                         color="gray", style="italic", multialignment="right")

        print(f"{tag} --- Sa interpolation vs PoE ---")
        interp_summary = []

        for imt_str in self.imts:
            period = self._imt_to_period(imt_str)
            if np.isnan(period):
                continue
            if period not in periods and (period != 0.01 or 0.01 not in periods):
                continue

            imt_idx   = self._imt_idx[imt_str]
            sa_values = np.array(self.imtls[imt_str])
            poe_vals  = self._hcurves_stats[0, stat_idx, imt_idx, :]
            inv_Tr    = calculate_inv_Tr_from_poes(poe_vals, N=N)
            label     = "PGA" if imt_str == "PGA" else f"SA({period:.2f})"
            is_pga    = imt_str == "PGA"

            ax1.plot(sa_values, poe_vals,
                     "-o" if is_pga else "-",
                     linewidth=1.8 if is_pga else 1.5,
                     color="red" if is_pga else None,
                     label=label)
            ax2.plot(sa_values, inv_Tr,
                     "-o" if is_pga else "-",
                     linewidth=1.8 if is_pga else 1.5,
                     color="red" if is_pga else None,
                     label=label)

            if reference_value:
                parts = []
                for val in reference_value:
                    sa_ref = interpolate_sa_at_reference(poe_vals, sa_values, val)
                    parts.append(f"{label}={sa_ref:.3f}g")
                print(f"{tag} PoE={val:.3f} -> {' | '.join(parts)}")
                interp_summary.extend(parts)

        if reference_value:
            for val in reference_value:
                ax1.axhline(val, color="black", linestyle="--",
                            linewidth=1.2, label=f"PoE={val:.2f}")
            for val, tr_val in zip(reference_value, ref_inv_Tr):
                ax2.axhline(tr_val, color="black", linestyle="--",
                            linewidth=1.2, label=f"Tr={1/tr_val:.0f}y")

        site_title = title or f"Mean Hazard Curves at ({self.latitude:.2f}, {self.longitude:.2f}) | vs30={self.vs30:.0f} m/s"

        for ax_obj, ylabel, ylim in (
            (ax1, "PoE in 50y",            (1e-4, 1.0)),
            (ax2, "Annual Rate Exceedance", (1e-5, 0.1)),
        ):
            ax_obj.set_yscale("log")
            ax_obj.set_xlim(0.01, 2.0)
            ax_obj.set_ylim(*ylim)
            ax_obj.set_xlabel("Spectral Acceleration [g]", fontweight="bold")
            ax_obj.set_ylabel(ylabel, fontweight="bold")
            ax_obj.grid(True, which="both", linestyle="--", alpha=0.5)
            ax_obj.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
            ax_obj.set_title(site_title, fontweight="bold")
            ax_obj.figure.subplots_adjust(right=0.75)

        # build return dataframe
        records = []
        for imt_str in self.imts:
            period = self._imt_to_period(imt_str)
            if np.isnan(period):
                continue
            if period not in periods and (period != 0.01 or 0.01 not in periods):
                continue
            imt_idx   = self._imt_idx[imt_str]
            sa_values = np.array(self.imtls[imt_str])
            poe_vals  = self._hcurves_stats[0, stat_idx, imt_idx, :]
            inv_Tr    = calculate_inv_Tr_from_poes(poe_vals, N=N)
            for sa, poe_v, inv_tr_v in zip(sa_values, poe_vals, inv_Tr):
                records.append({
                    "imt":    imt_str,
                    "period": period,
                    "Sa":     float(sa),
                    "poe":    float(poe_v),
                    "inv_Tr": float(inv_tr_v),
                })

        df_out = pd.DataFrame(records)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig1.savefig(f"{save_path}_hazardcurves_PoE.svg",            format="svg", bbox_inches="tight")
            fig1.savefig(f"{save_path}_hazardcurves_PoE.pdf",            format="pdf", bbox_inches="tight")
            fig2.savefig(f"{save_path}_hazardcurves_AnualExcedence.svg", format="svg", bbox_inches="tight")
            fig2.savefig(f"{save_path}_hazardcurves_AnualExcedence.pdf", format="pdf", bbox_inches="tight")
            print(f"{tag} Figures saved: {os.path.basename(save_path)}_hazardcurves_PoE.svg / "
                  f"_hazardcurves_AnualExcedence.svg")

            plt.show()

        plt.close("all")
        return df_out

    # def plot_location_map(self):
    #     """Return a folium map with the site location."""
    #     print(f"[PSHA_HDF5] plot_location_map | lat={self.latitude:.4f}, lon={self.longitude:.4f}")
    #     return plot_uhs_location_map(lat=self.latitude, lon=self.longitude)

    def plot_location_map(self):
        import folium
        """Return a folium map with the site location."""
        print(f"[PSHA_HDF5] plot_location_map | lat={self.latitude:.4f}, lon={self.longitude:.4f}")

        mapa = folium.Map(location=[self.latitude, self.longitude], zoom_start=10, tiles=None)

        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            attr="© OpenStreetMap contributors © CARTO",
            name="Base (Positron)",
            overlay=False,
            control=True
        ).add_to(mapa)

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri",
            name="Hillshade",
            overlay=True,
            control=True,
            opacity=0.35
        ).add_to(mapa)

        folium.Marker(
            location=[self.latitude, self.longitude],
            popup=f"PSHA Site | lat={self.latitude:.4f}, lon={self.longitude:.4f}",
            tooltip="PSHA Site",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(mapa)

        return mapa