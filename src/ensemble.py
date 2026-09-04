import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier


def get_model_probabilities(models, X):
    """Получение вероятностей от всех моделей"""
    lr_pipeline, knn_pipeline, dt_pipeline, rf_pipeline, \
    catboost_pipeline, xgb_pipeline = models[:6]
    dnn_model, preprocessor = models[6], models[7]
    
    proba_dict = {}
    
    # Sklearn модели
    proba_dict['LR'] = lr_pipeline.predict_proba(X)[:, 1]
    proba_dict['KNN'] = knn_pipeline.predict_proba(X)[:, 1]
    proba_dict['DT'] = dt_pipeline.predict_proba(X)[:, 1]
    proba_dict['RF'] = rf_pipeline.predict_proba(X)[:, 1]
    proba_dict['CatBoost'] = catboost_pipeline.predict_proba(X)[:, 1]
    proba_dict['XGBoost'] = xgb_pipeline.predict_proba(X)[:, 1]
    
    # DNN
    X_processed = preprocessor.transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    dnn_model.eval()
    with torch.no_grad():
        dnn_outputs = dnn_model(X_tensor)
        proba_dict['DNN'] = torch.sigmoid(dnn_outputs).numpy().ravel()
    
    return pd.DataFrame(proba_dict)


def simple_averaging(probabilities_df, test, output_name='Averaging_submission.csv'):
    """Простое усреднение"""
    avg_proba = probabilities_df.mean(axis=1)
    avg_predict = (avg_proba >= 0.5).astype(int)
    
    output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': avg_predict
    })
    output.to_csv(output_name, index=False)
    print(f'{output_name} создан.')
    return avg_predict


def weighted_averaging(probabilities_df, test, weights=None, output_name='Weighted_Averaging_submission.csv'):
    """Взвешенное усреднение"""
    if weights is None:
        weights = {
            'LR': 0.5, 'KNN': 0.3, 'DT': 0.4,
            'RF': 1.0, 'CatBoost': 0.2, 'XGBoost': 1.0,
            'DNN': 0.6
        }
    
    weighted_proba = sum(
        probabilities_df[name] * weight
        for name, weight in weights.items()
    ) / sum(weights.values())
    
    weighted_predict = (weighted_proba >= 0.5).astype(int)
    
    output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': weighted_predict
    })
    output.to_csv(output_name, index=False)
    print(f'{output_name} создан.')
    return weighted_predict


def soft_voting(models, X_train, y_train, X_kaggle, test, output_name='Voting_submission.csv'):
    """Soft Voting"""
    lr_pipeline, knn_pipeline, dt_pipeline, rf_pipeline, \
    catboost_pipeline, xgb_pipeline = models[:6]
    
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
    
    voting_proba = voting_clf.predict_proba(X_kaggle)[:, 1]
    voting_predict = (voting_proba >= 0.5).astype(int)
    
    output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': voting_predict
    })
    output.to_csv(output_name, index=False)
    print(f'{output_name} создан.')
    return voting_predict


def hard_voting(models, X_train, y_train, X_kaggle, test, output_name='Hard_Voting_submission.csv'):
    """Hard Voting"""
    lr_pipeline, knn_pipeline, dt_pipeline, rf_pipeline, \
    catboost_pipeline, xgb_pipeline = models[:6]
    
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
    
    hard_voting_predict = voting_hard.predict(X_kaggle).astype(int)
    
    output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': hard_voting_predict
    })
    output.to_csv(output_name, index=False)
    print(f'{output_name} создан.')
    return hard_voting_predict


def stacking_classifier(models, X_train, y_train, X_kaggle, test, 
                        final_estimator=None, output_name='Stacking_submission.csv'):
    """Stacking с произвольным final estimator"""
    lr_pipeline, knn_pipeline, dt_pipeline, rf_pipeline, \
    catboost_pipeline, xgb_pipeline = models[:6]
    
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000)
    
    stacking = StackingClassifier(
        estimators=[
            ('lr', lr_pipeline),
            ('knn', knn_pipeline),
            ('dt', dt_pipeline),
            ('rf', rf_pipeline),
            ('catboost', catboost_pipeline),
            ('xgb', xgb_pipeline)
        ],
        final_estimator=final_estimator,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    stacking.fit(X_train, y_train)
    
    # Для RidgeClassifier используем predict, для остальных - predict_proba
    if isinstance(final_estimator, RidgeClassifier):
        stacking_predict = stacking.predict(X_kaggle).astype(int)
    else:
        stacking_proba = stacking.predict_proba(X_kaggle)[:, 1]
        stacking_predict = (stacking_proba >= 0.5).astype(int)
    
    output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': stacking_predict
    })
    output.to_csv(output_name, index=False)
    print(f'{output_name} создан.')
    return stacking_predict


def create_all_ensembles(models, X_train, y_train, X_kaggle, test):
    """Создание всех ансамблей"""
    # Получаем вероятности
    proba_df = get_model_probabilities(models, X_kaggle)
    print('Первые вероятности моделей:')
    print(proba_df.head())
    
    # Все ансамбли
    averaging_pred = simple_averaging(proba_df, test, 'submissions/Averaging_submission.csv')
    weighted_pred = weighted_averaging(proba_df, test, 'submissions/Weighted_Averaging_submission.csv')
    soft_voting_pred = soft_voting(models, X_train, y_train, X_kaggle, test, 'submissions/Voting_submission.csv')
    hard_voting_pred = hard_voting(models, X_train, y_train, X_kaggle, test, 'submissions/Hard_Voting_submission.csv')
    stacking_lr_pred = stacking_classifier(
        models, X_train, y_train, X_kaggle, test,
        LogisticRegression(max_iter=1000),
        'submissions/Stacking_LogisticRegression_submission.csv'
    )
    stacking_ridge_pred = stacking_classifier(
        models, X_train, y_train, X_kaggle, test,
        RidgeClassifier(),
        'submissions/Stacking_Ridge_submission.csv'
    )
    
    # Сохраняем все предсказания в одну таблицу
    ensemble_predictions = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Averaging': averaging_pred,
        'Weighted_Averaging': weighted_pred,
        'Soft_Voting': soft_voting_pred,
        'Hard_Voting': hard_voting_pred,
        'Stacking_LR': stacking_lr_pred,
        'Stacking_Ridge': stacking_ridge_pred
    })
    
    ensemble_predictions.to_csv('submissions/All_Ensemble_predictions.csv', index=False)
    print('\nВсе ансамбли созданы.')
    
    return ensemble_predictions
