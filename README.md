# Dataset Comparison Tool

##  Description
This script compares two datasets by:
- Computing statistics (mean, std, variance, etc.)
- Modifying a dataset (with configurable noise and missing values)
- Comparing changes with respect to percentage thresholds (global and per-measurement)

All settings (paths, thresholds, probabilities...) are defined in a single YAML config file.

---
## Python Version

This project is compatible with: Python 3.11.5

##  How to Run

```install all dependencies:
pip install -r requirements.txt


```run the project:
``runs the entire pipeline:
python main.py --config config.yaml

``same as above, but explicitly using --mode all
python main.py --config config.yaml --mode all

``only modify datasets (no report generation)
python main.py --config config.yaml --mode stats

``only compare and generate report from existing files
python main.py --config config.yaml --mode compare

