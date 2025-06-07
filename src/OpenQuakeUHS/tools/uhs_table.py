import pandas as pd
from OpenQuakeUHS.core.spectrum_parser import UHSSpectrum
import re

def generate_uhs_table(mean_file, quantile_files, poe=0.1):
    """
    Generate a table of spectral acceleration for a given PoE, combining
    mean and quantile spectra into a single DataFrame.

    Parameters:
    -----------
    mean_file : str
        Path to the mean CSV file.
    quantile_files : list of str
        List of paths to quantile CSV files (e.g., q16, q50, q84).
    poe : float
        Probability of exceedance to extract.

    Returns:
    --------
    df : pandas.DataFrame
        Table with columns: 'T', 'mean', 'q16', 'q50', 'q84'
    """
    # Inicializar tabla con periodo y mean
    mean_uhs = UHSSpectrum(mean_file)
    T = mean_uhs.mean.T()
    data = {
        'T': T,
        'mean': mean_uhs.mean.Sa(poe)
    }

    # Agregar cada cuantil extraído desde nombre de archivo
    for file in quantile_files:
        uhs = UHSSpectrum(file)
        # extraer número del cuantil (ej: q16)
        match = re.search(r"[-_](0\.\d+)", file)
        if match:
            qval = int(round(float(match.group(1)) * 100))
            label = f"q{qval}"
            data[label] = uhs.mean.Sa(poe)

    # Crear DataFrame
    df = pd.DataFrame(data)
    return df
