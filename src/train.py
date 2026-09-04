import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessor import create_preprocessor


def train_logistic_regression(X_train, y_train):
    """Обучение Logistic Regression"""
    lr_pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', LogisticRegression(max_iter=1000))
    ])
    
    lr_param_grid = {
        'model__C': [1],
        'model__solver': ['lbfgs']
    }
    
    lr_grid = GridSearchCV(
        lr_pipeline, lr_param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    
    return lr_grid.best_estimator_


def train_knn(X_train, y_train):
    """Обучение KNN"""
    knn_pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', KNeighborsClassifier())
    ])
    
    knn_param_grid = {
        'model__n_neighbors': [7],
        'model__weights': ['uniform'],
        'model__metric': ['manhattan']
    }
    
    knn_grid = GridSearchCV(
        knn_pipeline, knn_param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    knn_grid.fit(X_train, y_train)
    
    return knn_grid.best_estimator_


def train_decision_tree(X_train, y_train):
    """Обучение Decision Tree"""
    dt_pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])
    
    dt_param_grid = {
        'model__max_depth': [10],
        'model__min_samples_split': [20],
        'model__min_samples_leaf': [2],
        'model__criterion': ['entropy']
    }
    
    dt_grid = GridSearchCV(
        dt_pipeline, dt_param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    dt_grid.fit(X_train, y_train)
    
    return dt_grid.best_estimator_


def train_random_forest(X_train, y_train):
    """Обучение Random Forest"""
    rf_pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    
    rf_param_grid = {
        'model__n_estimators': [300],
        'model__max_depth': [20],
        'model__min_samples_split': [2],
        'model__min_samples_leaf': [2],
        'model__max_features': ['sqrt']
    }
    
    rf_grid = GridSearchCV(
        rf_pipeline, rf_param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    
    return rf_grid.best_estimator_


def train_catboost(X_train, y_train):
    """Обучение CatBoost"""
    catboost_pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', CatBoostClassifier(
            loss_function='Logloss',
            verbose=False,
            random_seed=RANDOM_STATE
        ))
    ])
    
    catboost_param_grid = {
        'model__iterations': [100],
        'model__depth': [6],
        'model__learning_rate': [0.01]
    }
    
    catboost_grid = GridSearchCV(
        catboost_pipeline, catboost_param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    catboost_grid.fit(X_train, y_train)
    
    return catboost_grid.best_estimator_


def train_xgboost(X_train, y_train):
    """Обучение XGBoost"""
    xgb_pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE))
    ])
    
    xgb_param_grid = {
        'model__n_estimators': [300],
        'model__max_depth': [6],
        'model__learning_rate': [0.01],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.8]
    }
    
    xgb_grid = GridSearchCV(
        xgb_pipeline, xgb_param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train)
    
    return xgb_grid.best_estimator_


def cross_validate_model(model, X_train, y_train):
    """Кросс-валидация модели"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print('Accuracy на каждом фолде:')
    for i, score in enumerate(scores, 1):
        print(f'Fold {i}: {score:.4f}')
    
    print(f'\nСредняя accuracy: {scores.mean():.4f}')
    print(f'Стандартное отклонение: {scores.std():.4f}')
    
    return scores
