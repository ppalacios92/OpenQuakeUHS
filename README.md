📁 OpenQuakeUHS/
├── README.md
├── setup.py
├── pyproject.toml
├── 📁 src/
│ └── 📁 OpenQuakeUHS/
│ ├── 📁 config/
│ │ └── settings.py # Configuración general si es necesario
│ ├── 📁 core/
│ │ ├── spectrum_parser.py # Clases UHSSpectrum y UHSCurves
│ │ └── folder_classifier.py # Clasificación de archivos mean, rlz, quantile
│ ├── 📁 models/
│ │ └── uhs_result.py # Clases de alto nivel para representar resultados por sitio
│ ├── 📁 tools/
│ │ ├── uhs_plotter.py # Funciones para graficar múltiples espectros
│ │ └── uhs_table.py # Generación de tablas combinadas T, mean, q16, q50, q84
│ ├── 📁 examples/
│ │ ├── example_plot_all.ipynb # Notebook de uso
│ │ └── 📁 data/ # Carpeta de datos de prueba
├── 📁 OpenQuakeUHS.egg-info/ # Generado al instalar localmente
│ ├── PKG-INFO
│ ├── SOURCES.txt
│ ├── dependency_links.txt
│ └── top_level.txt