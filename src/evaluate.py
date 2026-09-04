import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from src.ensemble import get_model_probabilities


def evaluate_individual_models(models, X_valid, y_valid):
    """Оценка отдельных моделей"""
    lr_pipeline, knn_pipeline, dt_pipeline, rf_pipeline, \
    catboost_pipeline, xgb_pipeline = models[:6]
    dnn_model, preprocessor = models[6], models[7]
    
    # Получаем вероятности
    proba_df = get_model_probabilities(models, X_valid)
    
    # Оценка каждой модели
    validation_results = {}
    
    for model_name in proba_df.columns:
        prediction = (proba_df[model_name] >= 0.5).astype(int)
        validation_results[model_name] = accuracy_score(y_valid, prediction)
    
    return proba_df, validation_results


def evaluate_ensembles(proba_df, models, X_train, y_train, X_valid, y_valid):
    """Оценка всех ансамблей"""
    lr_pipeline, knn_pipeline, dt_pipeline, rf_pipeline, \
    catboost_pipeline, xgb_pipeline = models[:6]
    
    validation_results = {}
    
    # Averaging
    avg_proba = proba_df.mean(axis=1)
    avg_pred = (avg_proba >= 0.5).astype(int)
    validation_results['Averaging'] = accuracy_score(y_valid, avg_pred)
    
    # Weighted Averaging (с равными весами)
    weights = {name: 1.0 for name in proba_df.columns}
    weighted_proba = sum(proba_df[name] * weight for name, weight in weights.items()) / sum(weights.values())
    weighted_pred = (weighted_proba >= 0.5).astype(int)
    validation_results['Weighted Averaging'] = accuracy_score(y_valid, weighted_pred)
    
    # Soft Voting
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr_pipeline),
            ('knn', knn_pipeline),
            ('dt', dt_pipeline),
            ('rf', rf_pipeline),
            ('catboost', catboost_pipeline),
            ('xgb', xgb_pipeline)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    soft_voting_pred = voting_clf.predict(X_valid)
    validation_results['Soft Voting'] = accuracy_score(y_valid, soft_voting_pred)
    
    # Hard Voting
    voting_hard = VotingClassifier(
        estimators=[
            ('lr', lr_pipeline),
            ('knn', knn_pipeline),
            ('dt', dt_pipeline),
            ('rf', rf_pipeline),
            ('catboost', catboost_pipeline),
            ('xgb', xgb_pipeline)
        ],
        voting='hard'
    )
    voting_hard.fit(X_train, y_train)
    hard_voting_pred = voting_hard.predict(X_valid)
    validation_results['Hard Voting'] = accuracy_score(y_valid, hard_voting_pred)
    
    # Stacking LR
    stacking_lr = StackingClassifier(
        estimators=[
            ('lr', lr_pipeline),
            ('knn', knn_pipeline),
            ('dt', dt_pipeline),
            ('rf', rf_pipeline),
            ('catboost', catboost_pipeline),
            ('xgb', xgb_pipeline)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    stacking_lr.fit(X_train, y_train)
    stacking_lr_pred = stacking_lr.predict(X_valid)
    validation_results['Stacking LR'] = accuracy_score(y_valid, stacking_lr_pred)
    
    # Stacking Ridge
    stacking_ridge = StackingClassifier(
        estimators=[
            ('lr', lr_pipeline),
            ('knn', knn_pipeline),
            ('dt', dt_pipeline),
            ('rf', rf_pipeline),
            ('catboost', catboost_pipeline),
            ('xgb', xgb_pipeline)
        ],
        final_estimator=RidgeClassifier(),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    stacking_ridge.fit(X_train, y_train)
    stacking_ridge_pred = stacking_ridge.predict(X_valid)
    validation_results['Stacking Ridge'] = accuracy_score(y_valid, stacking_ridge_pred)
    
    return validation_results, voting_clf, voting_hard, stacking_lr, stacking_ridge


def get_best_model(validation_results_df):
    """Определение лучшей модели"""
    best_model = validation_results_df.iloc[0]['Model']
    best_score = validation_results_df.iloc[0]['Validation Accuracy']
    
    print(f'\nЛучший результат: {best_model}')
    print(f'Validation Accuracy: {best_score:.4f}')
    
    return best_model, best_score
