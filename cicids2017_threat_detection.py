# Source-code export from notebooks/ml_cybersecurity_cicids2017.ipynb
# Academic implementation inspired by RP1 and using CICIDS2017.


# ===== Notebook code cell 1 =====
import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

print('Python ML environment ready.')

# ===== Notebook code cell 2 =====
DATA_DIR_CANDIDATES = [
    './cicids2017',
    '/content/cicids2017',
    os.path.abspath('./cicids2017'),
    os.path.abspath('/content/cicids2017')
]

DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if os.path.isdir(p)), './cicids2017')
SAMPLE_SIZE = 250_000

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        'Could not find the CICIDS2017 CSV folder. Download or upload the dataset into one of these locations:\n'
        + '\n'.join(DATA_DIR_CANDIDATES)
        + '\n\nIf you are in Colab, run the data-download notebook first. If you are on a local machine, '
        'make sure the extracted CSV files are present under the selected folder.'
    )

print(f'Using dataset directory: {DATA_DIR}')

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '**', '*.csv'), recursive=True))
if not csv_files:
    raise FileNotFoundError(f'No CSV files found under {DATA_DIR}')

print(f'Found {len(csv_files)} CSV files.')
for f in csv_files:
    print(f' - {os.path.basename(f)}')

# ===== Notebook code cell 3 =====
def optimise_dtypes(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

parts = []
for f in csv_files:
    part = pd.read_csv(f, low_memory=False)
    part.columns = part.columns.str.strip()
    part['source_file'] = os.path.basename(f)
    parts.append(optimise_dtypes(part))
    print(f'Loaded {os.path.basename(f):55s} -> {part.shape}')

df = pd.concat(parts, ignore_index=True)
del parts

print(f'Combined shape: {df.shape}')

# ===== Notebook code cell 4 =====
# Clean label text and inspect the original attack classes.
if 'Label' not in df.columns:
    raise KeyError("Expected a 'Label' column in CICIDS2017.")

df['Label'] = df['Label'].astype(str).str.strip()

label_counts = df['Label'].value_counts()
print('Original label distribution:')
display(label_counts.to_frame('count'))

# ===== Notebook code cell 5 =====
# Stratified sampling for a practical Colab run.
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    sample = (
        df.groupby('Label', group_keys=False)
          .apply(lambda g: g.sample(
              n=min(len(g), max(1, int(round(SAMPLE_SIZE * len(g) / len(df))))),
              random_state=RANDOM_SEED
          ))
          .reset_index(drop=True)
    )
else:
    sample = df.copy()

del df

print(f'Sampled dataset shape: {sample.shape}')
print('Sampled labels:')
display(sample['Label'].value_counts().to_frame('count'))

# ===== Notebook code cell 6 =====
def map_security_category(label):
    s = label.lower()
    if s == 'benign':
        return 'normal'
    if 'bot' in s:
        return 'novel'
    return 'known_attack'

sample['security_category'] = sample['Label'].map(map_security_category)

print(sample['security_category'].value_counts())

novel_count = int((sample['security_category'] == 'novel').sum())
if novel_count < 50:
    raise ValueError(
        f'Only {novel_count} Bot samples were found. Increase SAMPLE_SIZE or use a larger dataset sample.'
    )

# ===== Notebook code cell 7 =====
# Keep the original label for interpretation, but exclude it from ML features.
known_df = sample[sample['security_category'] != 'novel'].copy()
novel_df = sample[sample['security_category'] == 'novel'].copy()

print('Known-data shape  :', known_df.shape)
print('Novel-data shape  :', novel_df.shape)
print('Novel attack label:', novel_df['Label'].value_counts().to_dict())

# ===== Notebook code cell 8 =====
meta_cols = {'Label', 'security_category', 'source_file'}
feature_candidates = [c for c in known_df.columns if c not in meta_cols]

# Convert candidate features to numeric and remove non-numeric metadata.
X_all = known_df[feature_candidates].apply(pd.to_numeric, errors='coerce')
X_novel_raw = novel_df[feature_candidates].apply(pd.to_numeric, errors='coerce')

# Replace infinities generated by rate features.
X_all = X_all.replace([np.inf, -np.inf], np.nan)
X_novel_raw = X_novel_raw.replace([np.inf, -np.inf], np.nan)

# Keep features with at least some observed values.
valid_cols = X_all.columns[X_all.notna().any()].tolist()
X_all = X_all[valid_cols]
X_novel_raw = X_novel_raw[valid_cols]

# Binary supervised target: normal vs known attack.
y_all = known_df['security_category'].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y_all
)

print('Initial numeric feature count:', len(valid_cols))
print('Training rows:', len(X_train_raw))
print('Test rows    :', len(X_test_raw))

# ===== Notebook code cell 9 =====
# Fit preprocessing only on the training set.
preprocess = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('variance', VarianceThreshold(threshold=0.0)),
    ('scaler', StandardScaler())
])

X_train = preprocess.fit_transform(X_train_raw)
X_test = preprocess.transform(X_test_raw)
X_novel = preprocess.transform(X_novel_raw)

print('Processed feature count:', X_train.shape[1])
print('X_train shape:', X_train.shape)
print('X_test shape :', X_test.shape)
print('X_novel shape:', X_novel.shape)

# ===== Notebook code cell 10 =====
models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=12,
        class_weight='balanced',
        random_state=RANDOM_SEED
    ),
    'Linear SVM': LinearSVC(
        class_weight='balanced',
        random_state=RANDOM_SEED,
        dual=False,
        max_iter=3000
    ),
    'MLP Neural Network': MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        early_stopping=True,
        validation_fraction=0.1,
        batch_size=512,
        max_iter=40,
        random_state=RANDOM_SEED
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=150,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
}

metrics_rows = []
trained_models = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    trained_models[name] = model
    metrics_rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, pos_label='known_attack'),
        'Recall': recall_score(y_test, preds, pos_label='known_attack'),
        'F1': f1_score(y_test, preds, pos_label='known_attack'),
        'Train time (s)': round(time.time() - t0, 1)
    })

results_supervised = pd.DataFrame(metrics_rows).sort_values('F1', ascending=False)
display(results_supervised)

# ===== Notebook code cell 11 =====
rf = trained_models['Random Forest']
rf_test_pred = rf.predict(X_test)

print('Random Forest classification report:')
print(classification_report(y_test, rf_test_pred, digits=4))

cm = confusion_matrix(y_test, rf_test_pred, labels=['normal', 'known_attack'])
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Normal', 'Known attack']
).plot(values_format='d')
plt.title('Random Forest — Known-threat classification')
plt.show()

# ===== Notebook code cell 12 =====
normal_train_mask = (y_train == 'normal')
X_normal_train = X_train[normal_train_mask]

# Cap the fitting sample for faster execution in Colab.
max_iforest_samples = 20_000
if len(X_normal_train) > max_iforest_samples:
    idx = rng.choice(len(X_normal_train), size=max_iforest_samples, replace=False)
    X_normal_fit = X_normal_train[idx]
else:
    X_normal_fit = X_normal_train

iso = IsolationForest(
    n_estimators=150,
    max_samples='auto',
    contamination=0.03,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

t0 = time.time()
iso.fit(X_normal_fit)
print(f'Isolation Forest trained on {len(X_normal_fit):,} normal samples in {time.time() - t0:.1f}s')

# ===== Notebook code cell 13 =====
X_test_normal = X_test[y_test == 'normal']

iso_novel_pred = iso.predict(X_novel)       # -1 = anomaly
iso_normal_pred = iso.predict(X_test_normal)

novel_catch = float((iso_novel_pred == -1).mean())
normal_false_alarm = float((iso_normal_pred == -1).mean())
rf_novel_pred = rf.predict(X_novel)
rf_novel_miss = float((rf_novel_pred == 'normal').mean())

unsupervised_results = pd.DataFrame({
    'Measure': [
        'Isolation Forest novel-attack catch rate',
        'Isolation Forest false-alarm rate on normal traffic',
        'Random Forest novel-attack miss rate'
    ],
    'Rate': [novel_catch, normal_false_alarm, rf_novel_miss]
})

display(unsupervised_results)

# ===== Notebook code cell 14 =====
plt.figure(figsize=(8, 4.5))
plt.bar(
    ['IF catch\nnovel Bot', 'IF false alarms\nnormal', 'RF misses\nnovel Bot'],
    [novel_catch, normal_false_alarm, rf_novel_miss]
)
plt.ylim(0, 1.05)
plt.ylabel('Rate')
plt.title('Known-threat model vs. anomaly detector on unseen Bot traffic')
plt.show()

# ===== Notebook code cell 15 =====
def hybrid_predict(X_scaled, supervised_model, anomaly_model):
    supervised_pred = supervised_model.predict(X_scaled)
    anomaly_pred = anomaly_model.predict(X_scaled)

    final = []
    for sup, anom in zip(supervised_pred, anomaly_pred):
        if sup == 'known_attack':
            final.append('known_attack')
        elif anom == -1:
            final.append('unknown_suspected')
        else:
            final.append('normal')
    return np.array(final)

# Evaluate on known test traffic + completely unseen Bot traffic.
X_eval = np.vstack([X_test, X_novel])
y_eval_true = np.concatenate([y_test, np.array(['novel'] * len(X_novel))])

baseline_pred = rf.predict(X_eval)
hybrid_pred = hybrid_predict(X_eval, rf, iso)

known_attack_mask = y_eval_true == 'known_attack'
novel_mask = y_eval_true == 'novel'
normal_mask = y_eval_true == 'normal'

baseline_known_recall = (baseline_pred[known_attack_mask] == 'known_attack').mean()
baseline_novel_catch = (baseline_pred[novel_mask] != 'normal').mean()
hybrid_known_recall = (hybrid_pred[known_attack_mask] == 'known_attack').mean()
hybrid_novel_catch = (hybrid_pred[novel_mask] != 'normal').mean()

baseline_false_alarm = (baseline_pred[normal_mask] != 'normal').mean()
hybrid_false_alarm = (hybrid_pred[normal_mask] != 'normal').mean()

hybrid_summary = pd.DataFrame({
    'Metric': [
        'Known attack recall',
        'Unseen Bot catch rate',
        'False-alarm rate on normal traffic'
    ],
    'Random Forest only': [baseline_known_recall, baseline_novel_catch, baseline_false_alarm],
    'Hybrid RF + Isolation Forest': [hybrid_known_recall, hybrid_novel_catch, hybrid_false_alarm]
})

display(hybrid_summary)

# ===== Notebook code cell 16 =====
plot_df = hybrid_summary.set_index('Metric')
ax = plot_df.plot(kind='bar', figsize=(9, 5))
ax.set_ylim(0, 1.05)
ax.set_ylabel('Rate')
ax.set_title('Hybrid detection compared with supervised-only detection')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

# ===== Notebook code cell 17 =====
RESPONSE_MAP = {
    'normal': 'No automated action. Continue normal monitoring.',
    'known_attack': 'Raise security alert and apply attack-specific investigation/mitigation.',
    'unknown_suspected': 'Quarantine or isolate the source/host for analyst review.'
}

# Show a mixed sample, including unseen Bot traffic.
sample_idx = rng.choice(len(y_eval_true), size=min(12, len(y_eval_true)), replace=False)

response_table = pd.DataFrame({
    'True category': y_eval_true[sample_idx],
    'Detector verdict': hybrid_pred[sample_idx],
    'Automated response': [RESPONSE_MAP[v] for v in hybrid_pred[sample_idx]]
})

display(response_table)

# ===== Notebook code cell 18 =====
# Optional: save the fitted supervised model and preprocessing pipeline for the project demo.
import joblib

joblib.dump(rf, 'rf_cicids2017_model.joblib')
joblib.dump(preprocess, 'cicids2017_preprocess.joblib')
joblib.dump(iso, 'isolation_forest_cicids2017.joblib')

print('Saved:')
print(' - rf_cicids2017_model.joblib')
print(' - cicids2017_preprocess.joblib')
print(' - isolation_forest_cicids2017.joblib')
