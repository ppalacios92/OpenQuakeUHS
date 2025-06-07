"""
File: folder_classifier.py
Author: Ing. Patricio Palacios Msc.
Date: June 6, 2025

Description:
------------
This module provides utilities to walk through a directory and classify
OpenQuake CSV outputs related to Uniform Hazard Spectra (UHS) into:

- Mean curves
- Realization curves (rlz)
- Quantile curves (e.g., 16th, 84th percentiles)

This classification supports batch loading and plotting operations.
"""

import os

def classify_csv_files(folder_path):
    """
    Walks through the given folder recursively and classifies all CSV files
    into three categories: mean, rlz, and quantile.

    Parameters:
    -----------
    folder_path : str
        Path to the directory containing UHS CSV files exported by OpenQuake.

    Returns:
    --------
    mean_files : list of str
        Paths to CSV files containing 'mean' in their filename.

    rlz_files : list of str
        Paths to CSV files containing 'rlz' in their filename.

    quantile_files : list of str
        Paths to CSV files starting with 'quantile' (case-insensitive).
    """
    mean_files = []
    rlz_files = []
    quantile_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(root, file)
                filename_lower = file.lower()

                if "mean" in filename_lower:
                    mean_files.append(file_path)
                elif "rlz" in filename_lower:
                    rlz_files.append(file_path)
                elif filename_lower.startswith("quantile"):
                    quantile_files.append(file_path)

    return mean_files, rlz_files, quantile_files
