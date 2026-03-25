"""
=============================================================================
  SEATTLE WEATHER PREDICTION PROJECT
  AI/ML Final Year Internship Project
  Dataset: seattle-weather.csv (1461 records, 2012-2015)
=============================================================================

  FEATURES  : precipitation, temp_max, temp_min, wind
  TARGET    : weather  (drizzle | rain | fog | snow | sun)
  TASK      : Multi-class Classification

  MODELS USED:
    1. Logistic Regression       (Baseline)
    2. Random Forest Classifier  (Ensemble)
    3. Gradient Boosting (XGBoost-style via sklearn)
    4. Support Vector Machine (SVM)
    5. K-Nearest Neighbors (KNN)

  INCLUDES:
    - EDA (Exploratory Data Analysis)
    - Feature Engineering
    - Model Training & Tuning
    - Cross-Validation
    - Confusion Matrix
    - Classification Report
    - Feature Importance
    - Accuracy Comparison Chart
    - Prediction on new/custom input
=============================================================================
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

import os
import sys

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "seattle-weather.csv"      # <-- change path if needed
RANDOM_SEED = 42
TEST_SIZE   = 0.20
PLOT_DIR    = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2.  LOAD DATA
# ─────────────────────────────────────────────
print("=" * 65)
print("  SEATTLE WEATHER PREDICTION  –  AI/ML Final Year Project")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

print(f"\n✔  Loaded dataset  →  {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ─────────────────────────────────────────────
# 3.  EDA  –  Exploratory Data Analysis
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("  SECTION 1 : EDA")
print("─" * 50)

print("\nDataset Info:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['weather'].value_counts())

# ── EDA Plot 1: Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EDA – Target Variable: Weather", fontsize=14, fontweight='bold')

vc = df['weather'].value_counts()
axes[0].bar(vc.index, vc.values, color=sns.color_palette("Set2", len(vc)))
axes[0].set_title("Weather Class Frequency")
axes[0].set_xlabel("Weather Type")
axes[0].set_ylabel("Count")
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].pie(vc.values, labels=vc.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set2", len(vc)), startangle=140)
axes[1].set_title("Weather Class Proportion")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_class_distribution.png", dpi=150)
plt.close()
print(f"\n  [Saved] {PLOT_DIR}/01_class_distribution.png")

# ── EDA Plot 2: Feature distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EDA – Feature Distributions", fontsize=14, fontweight='bold')
features = ['precipitation', 'temp_max', 'temp_min', 'wind']
colors   = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4']

for ax, feat, col in zip(axes.flat, features, colors):
    ax.hist(df[feat], bins=30, color=col, edgecolor='white', alpha=0.85)
    ax.set_title(f"Distribution of {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_feature_distributions.png", dpi=150)
plt.close()
print(f"  [Saved] {PLOT_DIR}/02_feature_distributions.png")

# ── EDA Plot 3: Feature vs Weather (box plots)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("EDA – Features vs Weather Type", fontsize=14, fontweight='bold')

for ax, feat in zip(axes.flat, features):
    groups = [df[df['weather'] == w][feat].values for w in vc.index]
    ax.boxplot(groups, labels=vc.index, patch_artist=True,
               boxprops=dict(facecolor='#74b9ff', alpha=0.7))
    ax.set_title(f"{feat} by Weather")
    ax.set_ylabel(feat)
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_features_vs_weather.png", dpi=150)
plt.close()
print(f"  [Saved] {PLOT_DIR}/03_features_vs_weather.png")

# ── EDA Plot 4: Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()
print(f"  [Saved] {PLOT_DIR}/04_correlation_heatmap.png")

# ── EDA Plot 5: Monthly avg temp trend
df['month'] = df['date'].dt.month
df['year']  = df['date'].dt.year

monthly = df.groupby('month')[['temp_max', 'temp_min']].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly.index, monthly['temp_max'], 'o-', color='#e17055',
        linewidth=2, label='Avg Max Temp')
ax.plot(monthly.index, monthly['temp_min'], 's-', color='#0984e3',
        linewidth=2, label='Avg Min Temp')
ax.fill_between(monthly.index, monthly['temp_min'], monthly['temp_max'],
                alpha=0.2, color='gray')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title("Monthly Avg Temperature Trend (Seattle)", fontsize=13, fontweight='bold')
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_monthly_temp_trend.png", dpi=150)
plt.close()
print(f"  [Saved] {PLOT_DIR}/05_monthly_temp_trend.png")

# ─────────────────────────────────────────────
# 4.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("  SECTION 2 : FEATURE ENGINEERING")
print("─" * 50)

df['temp_range']  = df['temp_max'] - df['temp_min']          # Diurnal range
df['temp_avg']    = (df['temp_max'] + df['temp_min']) / 2    # Mean temp
df['season']      = df['month'].map({                        # Season label
    1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring',
    5:'Spring', 6:'Summer', 7:'Summer', 8:'Summer',
    9:'Fall',   10:'Fall',  11:'Fall',  12:'Winter'
})
df['is_rainy_season'] = df['month'].isin([10,11,12,1,2,3]).astype(int)
df['high_wind']   = (df['wind'] > df['wind'].quantile(0.75)).astype(int)
df['heavy_rain']  = (df['precipitation'] > df['precipitation'].quantile(0.75)).astype(int)

season_enc = LabelEncoder()
df['season_enc'] = season_enc.fit_transform(df['season'])

feature_cols = [
    'precipitation', 'temp_max', 'temp_min', 'wind',
    'temp_range', 'temp_avg', 'month', 'season_enc',
    'is_rainy_season', 'high_wind', 'heavy_rain'
]
print(f"\n✔  Total features after engineering: {len(feature_cols)}")
print(f"   Features: {feature_cols}")

# ─────────────────────────────────────────────
# 5.  PREPARE DATA FOR MODELLING
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("  SECTION 3 : DATA PREPARATION")
print("─" * 50)

X = df[feature_cols]
y = df['weather']

le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    stratify=y_enc
)
print(f"\n✔  Train size : {X_train.shape[0]} samples")
print(f"✔  Test  size : {X_test.shape[0]} samples")
print(f"✔  Classes    : {list(class_names)}")

# ─────────────────────────────────────────────
# 6.  DEFINE MODELS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                      class_weight='balanced'))
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(n_estimators=200, max_depth=None,
                                          random_state=RANDOM_SEED,
                                          class_weight='balanced'))
    ]),
    "Gradient Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                              max_depth=5, random_state=RANDOM_SEED))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    SVC(kernel='rbf', C=10, gamma='scale',
                       random_state=RANDOM_SEED, class_weight='balanced'))
    ]),
    "KNN": Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    KNeighborsClassifier(n_neighbors=7, metric='minkowski'))
    ]),
}

# ─────────────────────────────────────────────
# 7.  TRAIN  &  CROSS-VALIDATE ALL MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("  SECTION 4 : MODEL TRAINING & CROSS-VALIDATION")
print("─" * 50)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
results = {}

for name, pipeline in models.items():
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=cv, scoring='accuracy', n_jobs=-1)
    # Train on full training set
    pipeline.fit(X_train, y_train)
    y_pred    = pipeline.predict(X_test)
    test_acc  = accuracy_score(y_test, y_pred)

    results[name] = {
        'pipeline':  pipeline,
        'cv_mean':   cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'test_acc':  test_acc,
        'y_pred':    y_pred,
    }
    print(f"\n  [{name}]")
    print(f"    CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test Accuracy : {test_acc:.4f}")

# ─────────────────────────────────────────────
# 8.  DETAILED REPORTS
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("  SECTION 5 : DETAILED CLASSIFICATION REPORTS")
print("─" * 50)

for name, res in results.items():
    print(f"\n{'='*55}")
    print(f"  {name.upper()}")
    print(f"{'='*55}")
    print(classification_report(
        y_test, res['y_pred'],
        target_names=class_names,
        zero_division=0
    ))

# ─────────────────────────────────────────────
# 9.  CONFUSION MATRICES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Confusion Matrices – All Models", fontsize=15, fontweight='bold')

for ax, (name, res) in zip(axes.flat, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}\nAcc={res['test_acc']:.3f}", fontsize=11)
    ax.tick_params(axis='x', rotation=30)

axes.flat[-1].axis('off')   # hide empty 6th subplot
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_confusion_matrices.png", dpi=150)
plt.close()
print(f"\n  [Saved] {PLOT_DIR}/06_confusion_matrices.png")

# ─────────────────────────────────────────────
# 10.  MODEL COMPARISON CHART
# ─────────────────────────────────────────────
names     = list(results.keys())
cv_means  = [results[n]['cv_mean']  for n in names]
cv_stds   = [results[n]['cv_std']   for n in names]
test_accs = [results[n]['test_acc'] for n in names]

x     = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, cv_means,  width, label='CV Accuracy (5-fold)',
               color='#74b9ff', yerr=cv_stds, capsize=5, edgecolor='white')
bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy',
               color='#00b894', edgecolor='white')

ax.set_xlabel("Model")
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison – CV vs Test Accuracy", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylim(0.5, 1.05)
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.annotate(f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha='center', fontsize=8)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_model_comparison.png", dpi=150)
plt.close()
print(f"  [Saved] {PLOT_DIR}/07_model_comparison.png")

# ─────────────────────────────────────────────
# 11.  BEST MODEL & FEATURE IMPORTANCE
# ─────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['test_acc'])
print("\n" + "─" * 50)
print(f"  SECTION 6 : BEST MODEL  →  {best_name}")
print(f"              Test Accuracy = {results[best_name]['test_acc']:.4f}")
print("─" * 50)

best_pipeline = results[best_name]['pipeline']
clf = best_pipeline.named_steps['clf']

# Feature importance (works for tree-based; coefficients for LR)
if hasattr(clf, 'feature_importances_'):
    importances = clf.feature_importances_
    title_str = "Feature Importances"
elif hasattr(clf, 'coef_'):
    importances = np.abs(clf.coef_).mean(axis=0)
    title_str = "Avg |Coefficient| across Classes"
else:
    importances = None

if importances is not None:
    fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi_df)))
    bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                   color=colors_fi, edgecolor='white')
    ax.set_title(f"{best_name} – {title_str}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score")
    ax.grid(axis='x', alpha=0.3)

    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/08_feature_importance.png", dpi=150)
    plt.close()
    print(f"  [Saved] {PLOT_DIR}/08_feature_importance.png")

    print(f"\nTop 5 Features ({best_name}):")
    print(fi_df.sort_values('Importance', ascending=False).head(5).to_string(index=False))

# ─────────────────────────────────────────────
# 12.  LIVE PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_weather(
    precipitation: float,
    temp_max:      float,
    temp_min:      float,
    wind:          float,
    month:         int  = None,
    model_name:    str  = None
) -> None:
    """
    Predict Seattle weather given raw sensor readings.

    Parameters
    ----------
    precipitation : float  – mm of rain
    temp_max      : float  – max temperature (°C)
    temp_min      : float  – min temperature (°C)
    wind          : float  – wind speed (m/s)
    month         : int    – 1..12 (defaults to June)
    model_name    : str    – one of the model keys; uses best model if None
    """
    if month is None:
        month = 6

    # Reproduce feature engineering
    temp_range        = temp_max - temp_min
    temp_avg          = (temp_max + temp_min) / 2
    is_rainy_season   = int(month in [10, 11, 12, 1, 2, 3])
    high_wind         = int(wind > df['wind'].quantile(0.75))
    heavy_rain        = int(precipitation > df['precipitation'].quantile(0.75))
    season_map        = {1:'Winter',2:'Winter',3:'Spring',4:'Spring',
                         5:'Spring',6:'Summer',7:'Summer',8:'Summer',
                         9:'Fall',10:'Fall',11:'Fall',12:'Winter'}
    season_label      = season_map[month]
    season_e          = season_enc.transform([season_label])[0]

    row = pd.DataFrame([[
        precipitation, temp_max, temp_min, wind,
        temp_range, temp_avg, month, season_e,
        is_rainy_season, high_wind, heavy_rain
    ]], columns=feature_cols)

    # Choose model
    mname   = model_name or best_name
    pipe    = results[mname]['pipeline']
    label   = pipe.predict(row)[0]
    weather = le.inverse_transform([label])[0]

    # Probabilities (if model supports predict_proba)
    clf_step = pipe.named_steps['clf']
    if hasattr(clf_step, 'predict_proba'):
        proba = pipe.predict_proba(row)[0]
        proba_str = "  Probabilities:\n"
        for cls, p in sorted(zip(class_names, proba), key=lambda x: -x[1]):
            bar = "█" * int(p * 20)
            proba_str += f"    {cls:<10} {bar:<20}  {p:.3f}\n"
    else:
        proba_str = "  (Model does not support probability output)\n"

    print(f"\n{'='*55}")
    print(f"  PREDICTION  [{mname}]")
    print(f"{'='*55}")
    print(f"  Input  : precipitation={precipitation} mm  |  "
          f"temp_max={temp_max}°C  |  temp_min={temp_min}°C  |  "
          f"wind={wind} m/s  |  month={month} ({season_label})")
    print(f"\n  ➤  Predicted Weather  :  {weather.upper()}")
    print()
    print(proba_str)

# ─────────────────────────────────────────────
# 13.  EXAMPLE PREDICTIONS
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("  SECTION 7 : EXAMPLE PREDICTIONS")
print("─" * 50)

predict_weather(precipitation=0.0,  temp_max=28.5, temp_min=15.0, wind=2.5, month=7)
predict_weather(precipitation=15.2, temp_max=8.0,  temp_min=3.0,  wind=5.0, month=1)
predict_weather(precipitation=0.5,  temp_max=12.0, temp_min=7.0,  wind=3.0, month=11)
predict_weather(precipitation=0.0,  temp_max=-1.0, temp_min=-5.0, wind=2.0, month=2)

# ─────────────────────────────────────────────
# 14.  FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 65)
print(f"  {'Model':<25} {'CV Accuracy':>12}  {'Std':>8}  {'Test Acc':>10}")
print("  " + "-" * 60)
for name in sorted(results, key=lambda n: results[n]['test_acc'], reverse=True):
    r = results[name]
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<25} {r['cv_mean']:>11.4f}  {r['cv_std']:>8.4f}  "
          f"{r['test_acc']:>10.4f}{marker}")

print("\n  All plots saved to →", os.path.abspath(PLOT_DIR))
print("\n  Project Complete ✔")
print("=" * 65)
