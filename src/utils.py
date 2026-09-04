import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance


def analyze_constant_features(X):
    """Анализ константных признаков"""
    print('Количество уникальных значений:')
    print(X.nunique().sort_values())
    
    constant_cols = X.columns[X.nunique() == 1].tolist()
    print('\nКонстантные признаки:')
    print(constant_cols)
    
    return constant_cols


def plot_correlation_matrix(X, num_cols, figsize=(8, 6)):
    """Построение матрицы корреляций"""
    corr = X[num_cols].corr()
    print(corr.round(2))
    
    plt.figure(figsize=figsize)
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.colorbar(label='Correlation')
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.show()
    
    return corr


def detect_outliers(X, num_cols):
    """Обнаружение выбросов по IQR"""
    outlier_counts = {}
    
    for col in num_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = ((X[col] < lower) | (X[col] > upper)).sum()
        outlier_counts[col] = outliers
        
        print(f'{col}: {outliers} выбросов ({outliers / len(X) * 100:.2f}%)')
    
    return outlier_counts


def plot_boxplots(X, num_cols, figsize=(6, 3)):
    """Построение boxplot'ов для числовых признаков"""
    for col in num_cols:
        plt.figure(figsize=figsize)
        plt.boxplot(X[col].dropna())
        plt.title(f'Boxplot: {col}')
        plt.ylabel(col)
        plt.show()


def feature_importance_rf(model, X_valid, y_valid, feature_names):
    """Важность признаков через permutation importance"""
    result = permutation_importance(
        model, X_valid, y_valid,
        scoring='accuracy',
        n_repeats=10,
        random_state=0,
        n_jobs=-1
    )
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    importance = importance.sort_values('importance_mean', ascending=False)
    print(importance.to_string(index=False))
    
    return importance
