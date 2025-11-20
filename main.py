import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

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
    lags = [1, 3, 6]
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
    df = df.iloc[24:].copy()
    return df

def prepare_train_test_data(df, target_col, horizon):
    df = df.copy()
    df['target'] = df[target_col].shift(-horizon)
    df = df.dropna(subset=['target'])
    exclude_cols = ['target', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    df[feature_cols] = df[feature_cols].ffill().bfill()
    years = df.index.year.unique()
    if len(years) > 1:
        train = df[df.index.year == 2004].copy()
        test = df[df.index.year == 2005].copy()
    else:
        split_idx = int(len(df) * 0.7)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]
    y_test = test['target']
    train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    if len(X_test) == 0:
        raise ValueError(f"Test set is empty!")
    if len(X_train) == 0:
        raise ValueError(f"Train set is empty!")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def preprocess_features(X_train, X_test):
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler

def train_mlp_regression(X_train, y_train, X_test, y_test, horizon, target_col):
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', 
                       max_iter=500, random_state=42, early_stopping=True, 
                       validation_fraction=0.1, verbose=False)
    mlp.fit(X_train, y_train)
    y_pred_test = mlp.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    naive_pred = np.full(len(y_test), y_train.mean())
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    return {
        'model': mlp,
        'predictions': y_pred_test,
        'y_test': y_test,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'naive_rmse': naive_rmse,
        'test_index': y_test.index
    }

def train_xgboost_regression(X_train, y_train, X_test, y_test, pollutant_name, horizon):
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    naive_col = f'{pollutant_name}_lag_{horizon}'
    if naive_col in X_test.columns:
        y_naive = X_test[naive_col].values
        rmse_naive = np.sqrt(mean_squared_error(y_test, y_naive))
        improvement = ((rmse_naive - rmse) / rmse_naive) * 100
    else:
        rmse_naive = None
        improvement = None
    metrics = {
        'pollutant': pollutant_name,
        'horizon': horizon,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rmse_naive': rmse_naive,
        'improvement_%': improvement
    }
    return model, y_pred, metrics

def train_rf_classification(X_train, y_train, X_test, y_test, horizon, thresholds=[1.5, 2.5]):
    def discretize(y):
        return pd.cut(y, bins=[-np.inf] + thresholds + [np.inf], labels=['Low', 'Mid', 'High'])
    y_train_class = discretize(y_train)
    y_test_class = discretize(y_test)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_class)
    y_pred_test = rf.predict(X_test)
    accuracy = accuracy_score(y_test_class, y_pred_test)
    precision = precision_score(y_test_class, y_pred_test, average='macro', zero_division=0)
    recall = recall_score(y_test_class, y_pred_test, average='macro', zero_division=0)
    f1 = f1_score(y_test_class, y_pred_test, average='macro', zero_division=0)
    naive_pred = [y_train_class.mode()[0]] * len(y_test_class)
    naive_accuracy = accuracy_score(y_test_class, naive_pred)
    cm = confusion_matrix(y_test_class, y_pred_test, labels=['Low', 'Mid', 'High'])
    return {
        'model': rf,
        'predictions': y_pred_test,
        'y_test_class': y_test_class,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'naive_accuracy': naive_accuracy,
        'confusion_matrix': cm,
        'test_index': y_test.index
    }

def discretize_co(y_values):
    categories = np.zeros(len(y_values), dtype=object)
    categories[y_values < 1.5] = 'low'
    categories[(y_values >= 1.5) & (y_values < 2.5)] = 'mid'
    categories[y_values >= 2.5] = 'high'
    return categories

def train_gradient_boosting_classification(X_train, y_train, X_test, y_test, horizon):
    y_train_cat = discretize_co(y_train.values)
    y_test_cat = discretize_co(y_test.values)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_cat)
    y_test_encoded = le.transform(y_test_cat)
    
    gb_classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_classifier.fit(X_train_scaled, y_train_encoded)
    
    y_pred_encoded = gb_classifier.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_encoded)
    
    accuracy = accuracy_score(y_test_cat, y_pred)
    precision = precision_score(y_test_cat, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_cat, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test_cat, y_pred, average='macro', zero_division=0)
    
    naive_pred = np.full(len(y_test_cat), y_train_cat[0])
    naive_accuracy = accuracy_score(y_test_cat, naive_pred)
    
    cm = confusion_matrix(y_test_cat, y_pred, labels=['low', 'mid', 'high'])
    
    return {
        'model': gb_classifier,
        'scaler': scaler,
        'label_encoder': le,
        'predictions': y_pred,
        'y_test_class': y_test_cat,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'naive_accuracy': naive_accuracy,
        'confusion_matrix': cm,
        'test_index': y_test.index
    }

def evaluate_mlp_regression(df, pollutants, horizons=[1, 6, 12, 24]):
    print("\nMLP REGRESSION:")
    results = {}
    for pollutant in pollutants:
        if pollutant not in df.columns:
            continue
        print(f"\n{pollutant}:")
        results[pollutant] = {}
        for horizon in horizons:
            try:
                X_train, X_test, y_train, y_test, _ = prepare_train_test_data(df, pollutant, horizon)
                result = train_mlp_regression(X_train, y_train, X_test, y_test, horizon, pollutant)
                results[pollutant][horizon] = result
                print(f"  {horizon}h: RMSE={result['rmse']:.4f}, R²={result['r2']:.4f}")
            except Exception as e:
                continue
    return results

def evaluate_xgboost_regression(df, pollutants, horizons=[1, 6, 12, 24]):
    print("\nXGBOOST REGRESSION:")
    results = {}
    for pollutant in pollutants:
        if pollutant not in df.columns:
            continue
        print(f"\n{pollutant}:")
        results[pollutant] = {}
        for horizon in horizons:
            try:
                X_train, X_test, y_train, y_test, _ = prepare_train_test_data(df, pollutant, horizon)
                model, y_pred, metrics = train_xgboost_regression(X_train, y_train, X_test, y_test, pollutant, horizon)
                results[pollutant][horizon] = {
                    'model': model,
                    'predictions': y_pred,
                    'y_test': y_test,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'naive_rmse': metrics['rmse_naive'],
                    'test_index': y_test.index
                }
                print(f"  {horizon}h: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            except Exception as e:
                continue
    return results

def evaluate_rf_classification(df, target_col='CO(GT)', horizons=[1, 6, 12, 24]):
    print("\nRANDOM FOREST CLASSIFICATION:")
    results = {}
    for horizon in horizons:
        try:
            X_train, X_test, y_train, y_test, _ = prepare_train_test_data(df, target_col, horizon)
            result = train_rf_classification(X_train, y_train, X_test, y_test, horizon)
            results[horizon] = result
            print(f"  {horizon}h: Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}")
        except Exception as e:
            continue
    return results

def evaluate_gb_classification(df, target_col='CO(GT)', horizons=[1, 6, 12, 24]):
    print("\nGRADIENT BOOSTING CLASSIFICATION:")
    results = {}
    for horizon in horizons:
        try:
            X_train, X_test, y_train, y_test, _ = prepare_train_test_data(df, target_col, horizon)
            result = train_gradient_boosting_classification(X_train, y_train, X_test, y_test, horizon)
            results[horizon] = result
            print(f"  {horizon}h: Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}")
        except Exception as e:
            continue
    return results

def visualize_regression_results(reg_results, output_dir='./figures/', model_name='MLP'):
    os.makedirs(output_dir, exist_ok=True)
    pollutants = list(reg_results.keys())
    horizons = sorted(list(reg_results[pollutants[0]].keys()))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for pollutant in pollutants:
        rmse_values = [reg_results[pollutant][h]['rmse'] for h in horizons]
        ax.plot(horizons, rmse_values, marker='o', label=pollutant, linewidth=2.5, markersize=8)
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'{model_name} Regression: RMSE vs Horizon for All Pollutants', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{model_name.lower()}_rmse_all_pollutants.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for pollutant in pollutants:
        r2_values = [reg_results[pollutant][h]['r2'] for h in horizons]
        ax.plot(horizons, r2_values, marker='s', label=pollutant, linewidth=2.5, markersize=8)
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'{model_name} Regression: R² vs Horizon for All Pollutants', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{model_name.lower()}_r2_all_pollutants.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    for pollutant in pollutants:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        for i, horizon in enumerate(horizons):
            y_test = reg_results[pollutant][horizon]['y_test']
            y_pred = reg_results[pollutant][horizon]['predictions']
            r2 = reg_results[pollutant][horizon]['r2']
            rmse = reg_results[pollutant][horizon]['rmse']
            axes[i].scatter(y_test, y_pred, alpha=0.4, s=15, color='blue')
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                        'r--', lw=2, label='Perfect prediction')
            axes[i].set_xlabel('Actual', fontsize=11)
            axes[i].set_ylabel('Predicted', fontsize=11)
            axes[i].set_title(f'{horizon}h Forecast (R²={r2:.3f}, RMSE={rmse:.3f})', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        plt.suptitle(f'{pollutant}: Predicted vs Actual ({model_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}{model_name.lower()}_scatter_{pollutant.replace("(", "").replace(")", "")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    for pollutant in pollutants:
        horizon = 24
        if horizon in reg_results[pollutant]:
            y_test = reg_results[pollutant][horizon]['y_test']
            y_pred = reg_results[pollutant][horizon]['predictions']
            test_index = reg_results[pollutant][horizon]['test_index']
            n_points = min(500, len(y_test))
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(test_index[:n_points], y_test[:n_points], label='Actual', linewidth=1.5, alpha=0.8)
            ax.plot(test_index[:n_points], y_pred[:n_points], label='Predicted', linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel(f'{pollutant} Concentration', fontsize=12)
            ax.set_title(f'{pollutant}: 24-Hour Forecast - Actual vs Predicted ({model_name})', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}{model_name.lower()}_timeseries_{pollutant.replace("(", "").replace(")", "")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    for pollutant in pollutants:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        for i, horizon in enumerate(horizons):
            y_test = reg_results[pollutant][horizon]['y_test']
            y_pred = reg_results[pollutant][horizon]['predictions']
            residuals = y_test - y_pred
            axes[i].scatter(y_pred, residuals, alpha=0.4, s=15, color='green')
            axes[i].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[i].set_xlabel('Predicted Values', fontsize=11)
            axes[i].set_ylabel('Residuals', fontsize=11)
            axes[i].set_title(f'{horizon}h Forecast Residuals', fontsize=12)
            axes[i].grid(True, alpha=0.3)
        plt.suptitle(f'{pollutant}: Residual Analysis ({model_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}{model_name.lower()}_residuals_{pollutant.replace("(", "").replace(")", "")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def visualize_classification_results(clf_results, output_dir='./figures/', model_name='RF'):
    os.makedirs(output_dir, exist_ok=True)
    horizons = sorted(clf_results.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    acc = [clf_results[h]['accuracy'] for h in horizons]
    naive_acc = [clf_results[h]['naive_accuracy'] for h in horizons]
    ax1.plot(horizons, acc, marker='o', label=model_name, linewidth=2.5, markersize=8, color='blue')
    ax1.plot(horizons, naive_acc, marker='s', label='Naive Baseline', linewidth=2, linestyle='--', markersize=8, color='red')
    ax1.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{model_name} Classification: Accuracy vs Horizon', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    f1_scores = [clf_results[h]['f1'] for h in horizons]
    ax2.plot(horizons, f1_scores, marker='o', label='F1 Score', linewidth=2.5, markersize=8, color='green')
    ax2.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax2.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax2.set_title(f'{model_name} Classification: F1 Score vs Horizon', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{model_name.lower()}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    precision = [clf_results[h]['precision'] for h in horizons]
    recall = [clf_results[h]['recall'] for h in horizons]
    ax.plot(horizons, precision, marker='o', label='Precision', linewidth=2.5, markersize=8)
    ax.plot(horizons, recall, marker='s', label='Recall', linewidth=2.5, markersize=8)
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name} Classification: Precision and Recall vs Horizon', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{model_name.lower()}_precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for i, horizon in enumerate(horizons):
        cm = clf_results[horizon]['confusion_matrix']
        im = axes[i].imshow(cm, cmap='Blues', aspect='auto')
        for j in range(3):
            for k in range(3):
                text_color = "white" if cm[j, k] > cm.max()/2 else "black"
                axes[i].text(k, j, str(cm[j, k]), ha="center", va="center", 
                           color=text_color, fontsize=14, fontweight='bold')
        axes[i].set_xticks([0, 1, 2])
        axes[i].set_yticks([0, 1, 2])
        if model_name == 'RF':
            axes[i].set_xticklabels(['Low', 'Mid', 'High'])
            axes[i].set_yticklabels(['Low', 'Mid', 'High'])
        else:
            axes[i].set_xticklabels(['low', 'mid', 'high'])
            axes[i].set_yticklabels(['low', 'mid', 'high'])
        axes[i].set_xlabel('Predicted', fontsize=11)
        axes[i].set_ylabel('Actual', fontsize=11)
        acc_val = clf_results[horizon]['accuracy']
        axes[i].set_title(f'{horizon}h Forecast (Accuracy={acc_val:.3f})', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.suptitle(f'CO(GT) Classification: Confusion Matrices ({model_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}{model_name.lower()}_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_tables(mlp_results, xgb_results, rf_results, gb_results, output_dir='./figures/'):
    os.makedirs(output_dir, exist_ok=True)
    pollutants = list(mlp_results.keys())
    horizons = sorted(list(mlp_results[pollutants[0]].keys()))
    
    mlp_rmse_data = {}
    for pollutant in pollutants:
        mlp_rmse_data[pollutant] = [mlp_results[pollutant][h]['rmse'] for h in horizons]
    mlp_rmse_df = pd.DataFrame(mlp_rmse_data, index=[f'{h}h' for h in horizons])
    mlp_rmse_df.to_csv(f'{output_dir}mlp_regression_rmse_summary.csv')
    
    mlp_r2_data = {}
    for pollutant in pollutants:
        mlp_r2_data[pollutant] = [mlp_results[pollutant][h]['r2'] for h in horizons]
    mlp_r2_df = pd.DataFrame(mlp_r2_data, index=[f'{h}h' for h in horizons])
    mlp_r2_df.to_csv(f'{output_dir}mlp_regression_r2_summary.csv')
    
    xgb_rmse_data = {}
    for pollutant in pollutants:
        xgb_rmse_data[pollutant] = [xgb_results[pollutant][h]['rmse'] for h in horizons]
    xgb_rmse_df = pd.DataFrame(xgb_rmse_data, index=[f'{h}h' for h in horizons])
    xgb_rmse_df.to_csv(f'{output_dir}xgboost_regression_rmse_summary.csv')
    
    xgb_r2_data = {}
    for pollutant in pollutants:
        xgb_r2_data[pollutant] = [xgb_results[pollutant][h]['r2'] for h in horizons]
    xgb_r2_df = pd.DataFrame(xgb_r2_data, index=[f'{h}h' for h in horizons])
    xgb_r2_df.to_csv(f'{output_dir}xgboost_regression_r2_summary.csv')
    
    rf_metrics = {
        'Accuracy': [rf_results[h]['accuracy'] for h in horizons],
        'Precision': [rf_results[h]['precision'] for h in horizons],
        'Recall': [rf_results[h]['recall'] for h in horizons],
        'F1 Score': [rf_results[h]['f1'] for h in horizons],
        'Naive Accuracy': [rf_results[h]['naive_accuracy'] for h in horizons]
    }
    rf_df = pd.DataFrame(rf_metrics, index=[f'{h}h' for h in horizons])
    rf_df.to_csv(f'{output_dir}rf_classification_summary.csv')
    
    gb_metrics = {
        'Accuracy': [gb_results[h]['accuracy'] for h in horizons],
        'Precision': [gb_results[h]['precision'] for h in horizons],
        'Recall': [gb_results[h]['recall'] for h in horizons],
        'F1 Score': [gb_results[h]['f1'] for h in horizons],
        'Naive Accuracy': [gb_results[h]['naive_accuracy'] for h in horizons]
    }
    gb_df = pd.DataFrame(gb_metrics, index=[f'{h}h' for h in horizons])
    gb_df.to_csv(f'{output_dir}gb_classification_summary.csv')

def main():
    DATA_FILE = 'AirQualityUCI.csv'
    POLLUTANTS = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    HORIZONS = [1, 6, 12, 24]
    OUTPUT_DIR = './figures/'
    
    df = load_and_preprocess_data(DATA_FILE)
    df = create_all_features(df)
    
    mlp_results = evaluate_mlp_regression(df, POLLUTANTS, HORIZONS)
    xgb_results = evaluate_xgboost_regression(df, POLLUTANTS, HORIZONS)
    rf_results = evaluate_rf_classification(df, 'CO(GT)', HORIZONS)
    gb_results = evaluate_gb_classification(df, 'CO(GT)', HORIZONS)
    
    visualize_regression_results(mlp_results, OUTPUT_DIR, 'MLP')
    visualize_regression_results(xgb_results, OUTPUT_DIR, 'XGBoost')
    visualize_classification_results(rf_results, OUTPUT_DIR, 'RF')
    visualize_classification_results(gb_results, OUTPUT_DIR, 'GB')
    create_summary_tables(mlp_results, xgb_results, rf_results, gb_results, OUTPUT_DIR)

if __name__ == "__main__":
    main()