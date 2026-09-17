# Machine Learning in Cybersecurity: Threat Detection and Response

Real-data ML implementation inspired by the research paper:

**Machine Learning in Cybersecurity: Threat Detection and Response**  
Tangevva Rudrappa Gadad, Varshitha R, Yoghana B K, and Yumlembam Henba Singh  
2025 1st International Conference on AIML-Applications for Engineering & Technology (ICAET), Pune, India, 16–17 January 2025.

## Project objective

Implement the cybersecurity threat-detection ideas from the paper using the **CICIDS2017** network-traffic benchmark.

The implementation combines:

- **Supervised learning:** Decision Tree, Linear SVM, MLP Neural Network, and Random Forest.
- **Unsupervised learning:** Isolation Forest trained on normal traffic.
- **Unseen-attack experiment:** Bot traffic is withheld from supervised training and used as a novel-attack test.
- **Hybrid detection:** Random Forest handles known attacks while Isolation Forest provides an anomaly-detection safety net.
- **Response simulation:** detection verdicts are mapped to simple incident-response actions.

## Repository structure

```text
ml-cybersecurity-threat-detection-cicids2017/
├── notebooks/
│   └── ml_cybersecurity_cicids2017.ipynb
├── src/
│   └── cicids2017_threat_detection.py
├── results/
│   └── README.md
├── figures/
│   └── README.md
├── models/
│   └── README.md
├── data/
│   └── README.md
├── docs/
│   └── report_notes.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project uses the CICIDS2017 benchmark dataset. **The raw CSV files are intentionally not included in this repository** because of their size and because the notebook is designed to obtain/use the dataset separately.

Place the extracted CSV files under:

```text
cicids2017/
```

or `/content/cicids2017/` in Google Colab.

See `data/README.md` for the expected setup.

## Running the implementation

The easiest route is Google Colab:

1. Open `notebooks/ml_cybersecurity_cicids2017.ipynb`.
2. Make sure CICIDS2017 CSV files are available under `/content/cicids2017/`.
3. Run the notebook from top to bottom.
4. Save the generated figures to `figures/`.
5. Save the trained `.joblib` files to `models/`.
6. Record the final metrics in `results/`.

A practical stratified sample is used so the implementation can run on a free Colab environment.

## Important experimental design

The supervised task is binary:

- `normal` = BENIGN traffic
- `known_attack` = attack types other than Bot
- `novel` = Bot traffic, completely withheld from supervised training

The anomaly detector is trained only on normal training traffic.

## Important limitation

CICIDS2017 is a benchmark dataset. Results from this academic implementation should not be interpreted as proof of equivalent performance on live production networks.

## GitHub publication

Suggested repository name:

`ml-cybersecurity-threat-detection-cicids2017`

After creating the GitHub repository, add its URL to the weekly progress report.

