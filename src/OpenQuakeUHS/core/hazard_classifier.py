"""
Hazard Curve File Classifier
Author: Ing. Patricio Palacios Msc.
Date: June 7, 2025

Description:
------------
This module provides utilities to classify OpenQuake hazard curve CSV files
into three categories:
- Mean curves
- Quantile curves (e.g., q16, q84)
- Realizations (rlz)

The classification is based on filename patterns.
"""

import os

def classify_hazard_files(folder_path):
    """
    Walks through the given folder recursively and classifies all CSV files
    into mean, quantile, and realization sets.

    Parameters:
    -----------
    folder_path : str
        Path to the directory containing hazard curve CSV files.

    Returns:
    --------
    mean_files : list of str
        Files that contain 'mean' in their name.

    quantile_files : list of str
        Files that start with 'quantile' or contain 'qXX'.

    rlz_files : list of str
        Files that contain 'rlz' in their name.
    """
    mean_files = []
    rlz_files = []
    quantile_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                path = os.path.join(root, file)
                name = file.lower()

                if "mean" in name:
                    mean_files.append(path)
                elif "rlz" in name:
                    rlz_files.append(path)
                elif name.startswith("quantile") or "q" in name:
                    quantile_files.append(path)

    return mean_files, rlz_files, quantile_files
