# OpenQuakeUHS

A Python-based toolkit for processing **Uniform Hazard Spectra (UHS)** and **seismic hazard disaggregation** results from OpenQuake outputs. This package is designed for structural engineers, researchers, and seismic modelers aiming to automate the interpretation and visualization of OpenQuake results using Python.

![Disaggregation SVG](src/OpenQuakeUHS/examples/DISS_disaggregation.svg)

---

## ⚙️ Features

- Automatically detects and loads OpenQuake CSV outputs for:
  - Epsilon disaggregation: `Mag_Dist_Eps-mean-*.csv`
  - TRT disaggregation: `TRT_Mag_Dist-mean-*.csv`
- Filters data by probability of exceedance (PoE) and intensity measure type (IMT).
- Calculates modal and mean values of magnitude and distance.
- Produces 3D disaggregation plots with:
  - Magnitude–Distance–Epsilon bar plots
  - Magnitude–Distance–TRT bar plots
- Built-in plotting utilities with `matplotlib` for publication-ready figures.
- Fully compatible with `pandas` for further statistical or graphical post-processing.

---

## 📦 Requirements

- Python 3.8 or higher
- Python libraries:
  - numpy
  - pandas
  - matplotlib

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ppalacios92/OpenQuakeUHS.git
cd OpenQuakeUHS
pip install -e .
```
---

## 📁 Repository Structure
```bash
OpenQuakeUHS/
├── core/ # Core logic for data loading and disaggregation handling
├── tools/ # Plotting and post-processing utilities
├── examples/ # Jupyter notebooks with usage examples
├── config/ # Optional color/style configuration files
├── tests/ # Unit tests for data extraction and plotting
└── README.md # Project documentation
```
---

## 🛑 Disclaimer

This tool is provided as-is, without any guarantees of accuracy, performance, or suitability for specific engineering tasks.
The author assumes no responsibility for the interpretation of results, post-processing errors, or consequences of incorrect usage.
Use at your own risk and always validate results against OpenQuake outputs and design codes.

---
## 👨‍💻 Author

Developed by Patricio Palacios B. Structural Engineer | Python Developer | Seismic Modeler GitHub: @ppalacios92
---
## 📚 How to Cite
If you use this tool in your work, please cite it as follows:
```bibtex
@misc{palacios2025openquakeuhs,
  author       = {Patricio Palacios B.},
  title        = {OpenQuakeUHS: A Python-based OpenQuake disaggregation and UHS visualization tool},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ppalacios92/OpenQuakeUHS}}
}
```

---

## 📄 Citation in APA (7th Edition)

Palacios B., P. (2025). OpenQuakeUHS: A Python-based OpenQuake disaggregation and UHS visualization tool [Computer software]. GitHub. https://github.com/ppalacios92/OpenQuakeUHS
