from setuptools import setup, find_packages

setup(
    name="OpenQuakeUHS",
    version="0.1.0",
    description="Parser and plotter for OpenQuake UHS outputs",
    author="Ing. Patricio Palacios Msc.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "matplotlib"
    ],
    python_requires=">=3.8",
)
