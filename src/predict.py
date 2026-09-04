import pandas as pd
import numpy as np
import torch

from src.dnn_model import mlp_predict


def generate_single_model_submissions(models, X_kaggle, test):
    """Генерация submission для отдельных моделей"""
    lr_pipeline, knn_pipeline, dt_pipeline, rf_pipeline, \
    catboost_pipeline, xgb_pipeline = models[:6]
    dnn_model, preprocessor = models[6], models[7]
    
    submissions = {}
    
    # Logistic Regression
    lr_predict = lr_pipeline.predict(X_kaggle)
    lr_output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': lr_predict
    })
    lr_output.to_csv('submissions/LR_submission.csv', index=False)
    submissions['LR'] = lr_predict
    
    # KNN
    knn_predict = knn_pipeline.predict(X_kaggle)
    knn_output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': knn_predict
    })
    knn_output.to_csv('submissions/KNN_submission.csv', index=False)
    submissions['KNN'] = knn_predict
    
    # Decision Tree
    dt_predict = dt_pipeline.predict(X_kaggle)
    dt_output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': dt_predict
    })
    dt_output.to_csv('submissions/DT_submission.csv', index=False)
    submissions['DT'] = dt_predict
    
    # Random Forest
    rf_predict = rf_pipeline.predict(X_kaggle)
    rf_output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': rf_predict
    })
    rf_output.to_csv('submissions/RF_submission.csv', index=False)
    submissions['RF'] = rf_predict
    
    # CatBoost
    catboost_predict = catboost_pipeline.predict(X_kaggle)
    catboost_output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': catboost_predict.astype(int).ravel()
    })
    catboost_output.to_csv('submissions/CatBoost_submission.csv', index=False)
    submissions['CatBoost'] = catboost_predict
    
    # XGBoost
    xgb_predict = xgb_pipeline.predict(X_kaggle)
    xgb_output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': xgb_predict.astype(int).ravel()
    })
    xgb_output.to_csv('submissions/XGB_submission.csv', index=False)
    submissions['XGBoost'] = xgb_predict
    
    # DNN
    dnn_predict = mlp_predict(dnn_model, X_kaggle, preprocessor)
    dnn_output = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': dnn_predict
    })
    dnn_output.to_csv('submissions/DNN_submission.csv', index=False)
    submissions['DNN'] = dnn_predict
    
    print('Все submission-файлы для отдельных моделей созданы.')
    return submissions
