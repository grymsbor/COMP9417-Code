import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_preprocess_data(filepath='AirQualityUCI.csv'):
    df = pd.read_csv(filepath, sep=';', decimal=',')
    df = df.replace(-200.0, np.nan)
    df = df.replace(-200, np.nan)
    try:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    except:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.set_index('DateTime')
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(how='all')
    df = df.ffill(limit=6)
    df = df.bfill()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"Data loaded: {len(df)} samples from {df.index.min()} to {df.index.max()}")
    return df

def create_all_features(df):
    pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    def get_season(month):
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3
    df['season'] = df['month'].apply(get_season)
    lags = [1, 3, 6]  # Reduced lag features to avoid too many NaN values
    for col in pollutants:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    windows = [3, 6, 12, 24]
    for col in pollutants:
        if col in df.columns:
            for window in windows:
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    
    # Drop first 24 rows to remove NaN values from lag features
    df = df.iloc[24:].copy()
    print(f"Features created: {len(df.columns)} total features, {len(df)} samples after cleaning")
    return df

def prepare_train_test_data(df, target_col, horizon):
    df = df.copy()
    df['target'] = df[target_col].shift(-horizon)
    
    # Remove rows with NaN in target first
    df = df.dropna(subset=['target'])
    
    exclude_cols = ['target', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Fill NaN values BEFORE splitting to ensure continuity across train/test boundary
    df[feature_cols] = df[feature_cols].ffill().bfill()
    
    # More conservative approach - use later split to ensure test set has data
    years = df.index.year.unique()
    if len(years) == 1:
        split_idx = int(len(df) * 0.7)  # Use 70/30 split instead of 75/25
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        print(f"Single year detected. Using 70/30 split: train={len(train)}, test={len(test)}")
    else:
        # Use 80% of time period for training to ensure enough test data
        total_days = (df.index.max() - df.index.min()).days
        train_days = int(total_days * 0.8)
        train_end = df.index.min() + pd.Timedelta(days=train_days)
        train = df[df.index <= train_end].copy()
        test = df[df.index > train_end].copy()
        print(f"Multi-year detected. Split at {train_end.strftime('%Y-%m-%d')}: train={len(train)}, test={len(test)}")
    
    # Extract features and targets
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols] 
    y_test = test['target']
    
    # Final NaN check and removal
    train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    print(f"After processing: train={len(X_train)}, test={len(X_test)}")
    
    if len(X_test) == 0:
        raise ValueError(f"Test set is empty! Check your data date range and split logic.")
    if len(X_train) == 0:
        raise ValueError(f"Train set is empty! Check your data.")
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_mlp_regression(X_train, y_train, X_test, y_test, horizon):
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1, verbose=False)
    mlp.fit(X_train, y_train)
    y_pred_test = mlp.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    naive_pred = X_test.iloc[:, 0].values
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    print(f"  {horizon}h - RMSE: {rmse:.4f}, R2: {r2:.4f}, Improvement: {((naive_rmse-rmse)/naive_rmse*100):.1f}%")
    return {'model': mlp, 'predictions': y_pred_test, 'y_test': y_test, 'rmse': rmse, 'mae': mae, 'r2': r2, 'naive_rmse': naive_rmse}

def train_rf_classification(X_train, y_train, X_test, y_test, horizon, thresholds=[1.5, 2.5]):
    def discretize(y):
        return pd.cut(y, bins=[-np.inf] + thresholds + [np.inf], labels=['Low', 'Mid', 'High'])
    y_train_class = discretize(y_train)
    y_test_class = discretize(y_test)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_class)
    y_pred_test = rf.predict(X_test)
    accuracy = accuracy_score(y_test_class, y_pred_test)
    precision = precision_score(y_test_class, y_pred_test, average='macro', zero_division=0)
    recall = recall_score(y_test_class, y_pred_test, average='macro', zero_division=0)
    f1 = f1_score(y_test_class, y_pred_test, average='macro', zero_division=0)
    naive_pred = [y_train_class.mode()[0]] * len(y_test_class)
    naive_accuracy = accuracy_score(y_test_class, naive_pred)
    cm = confusion_matrix(y_test_class, y_pred_test, labels=['Low', 'Mid', 'High'])
    print(f"  {horizon}h - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Improvement: {((accuracy-naive_accuracy)/naive_accuracy*100):.1f}%")
    return {'model': rf, 'predictions': y_pred_test, 'y_test_class': y_test_class, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'naive_accuracy': naive_accuracy, 'confusion_matrix': cm}

def evaluate_regression_all_horizons(df, target_col='CO(GT)', horizons=[1, 6, 12, 24]):
    print("\nMLP REGRESSION:")
    results = {}
    for horizon in horizons:
        X_train, X_test, y_train, y_test, _ = prepare_train_test_data(df, target_col, horizon)
        results[horizon] = train_mlp_regression(X_train, y_train, X_test, y_test, horizon)
    return results

def evaluate_classification_all_horizons(df, target_col='CO(GT)', horizons=[1, 6, 12, 24]):
    print("\nRANDOM FOREST CLASSIFICATION:")
    results = {}
    for horizon in horizons:
        X_train, X_test, y_train, y_test, _ = prepare_train_test_data(df, target_col, horizon)
        results[horizon] = train_rf_classification(X_train, y_train, X_test, y_test, horizon)
    return results

def visualize_regression_results(reg_results, target_col='CO(GT)', output_dir='./figures/'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    horizons = sorted(reg_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    mlp_rmse = [reg_results[h]['rmse'] for h in horizons]
    naive_rmse = [reg_results[h]['naive_rmse'] for h in horizons]
    ax.plot(horizons, mlp_rmse, marker='o', label='MLP', linewidth=2.5, markersize=8)
    ax.plot(horizons, naive_rmse, marker='s', label='Naive', linewidth=2, linestyle='--', markersize=8)
    ax.set_xlabel('Forecast Horizon (hours)')
    ax.set_ylabel('RMSE')
    ax.set_title(f'MLP Regression: RMSE vs Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}mlp_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for i, horizon in enumerate(horizons):
        y_test = reg_results[horizon]['y_test']
        y_pred = reg_results[horizon]['predictions']
        r2 = reg_results[horizon]['r2']
        axes[i].scatter(y_test, y_pred, alpha=0.4, s=15)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{horizon}h (R²={r2:.3f})')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}mlp_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figures to {output_dir}")

def visualize_classification_results(clf_results, target_col='CO(GT)', output_dir='./figures/'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    horizons = sorted(clf_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    rf_acc = [clf_results[h]['accuracy'] for h in horizons]
    naive_acc = [clf_results[h]['naive_accuracy'] for h in horizons]
    ax.plot(horizons, rf_acc, marker='o', label='RF', linewidth=2.5, markersize=8)
    ax.plot(horizons, naive_acc, marker='s', label='Naive', linewidth=2, linestyle='--', markersize=8)
    ax.set_xlabel('Forecast Horizon (hours)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'RF Classification: Accuracy vs Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}rf_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for i, horizon in enumerate(horizons):
        cm = clf_results[horizon]['confusion_matrix']
        im = axes[i].imshow(cm, cmap='Blues')
        for j in range(3):
            for k in range(3):
                axes[i].text(k, j, str(cm[j, k]), ha="center", va="center", color="black" if cm[j, k] < cm.max()/2 else "white")
        axes[i].set_xticks([0, 1, 2])
        axes[i].set_yticks([0, 1, 2])
        axes[i].set_xticklabels(['Low', 'Mid', 'High'])
        axes[i].set_yticklabels(['Low', 'Mid', 'High'])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_title(f'{horizon}h')
    plt.tight_layout()
    plt.savefig(f'{output_dir}rf_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figures to {output_dir}")

def main():
    DATA_FILE = 'AirQualityUCI.csv'
    TARGET_COL = 'CO(GT)'
    HORIZONS = [1, 6, 12, 24]
    df = load_and_preprocess_data(DATA_FILE)
    df = create_all_features(df)
    regression_results = evaluate_regression_all_horizons(df, TARGET_COL, HORIZONS)
    visualize_regression_results(regression_results, TARGET_COL)
    classification_results = evaluate_classification_all_horizons(df, TARGET_COL, HORIZONS)
    visualize_classification_results(classification_results, TARGET_COL)
    print("\n=== FINAL RESULTS ===")
    print("\nMLP REGRESSION (RMSE):")
    for h in HORIZONS:
        print(f"  {h}h: {regression_results[h]['rmse']:.4f}")
    print("\nRF CLASSIFICATION (Accuracy):")
    for h in HORIZONS:
        print(f"  {h}h: {classification_results[h]['accuracy']:.4f}")

if __name__ == "__main__":
    main()