import os
import sys
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data, get_features_target
from src.feature_engineering import engineer_features
from src.train import *
from src.dnn_model import prepare_dnn_data, train_dnn
from src.evaluate import *
from src.ensemble import create_all_ensembles
from src.predict import generate_single_model_submissions
from src.utils import analyze_constant_features, plot_correlation_matrix
from src.config import NUM_COLS, RANDOM_STATE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    print("=" * 80)
    print("🚢 TITANIC SURVIVAL PREDICTION PIPELINE")
    print("=" * 80)
    
    # 1. ЗАГРУЗКА ДАННЫХ
    print("\n📂 Загрузка данных...")
    train, test = load_data()
    X, y = get_features_target(train)
    
    print(f"   Train shape: {X.shape}")
    print(f"   Test shape: {test.shape}")
    
    # 2. FEATURE ENGINEERING
    print("\n🔧 Создание признаков...")
    X, test = engineer_features(X, test)
    X_kaggle = test.drop('PassengerId', axis=1)
    
    print(f"   После создания признаков: {X.shape}")
    
    # 3. АНАЛИЗ ДАННЫХ
    print("\n📊 Анализ данных...")
    constant_cols = analyze_constant_features(X)
    
    print("\n🔍 Корреляционный анализ...")
    plot_correlation_matrix(X, NUM_COLS)
    
    # 4. РАЗДЕЛЕНИЕ НА TRAIN/VALID
    print("\n📊 Разделение на train/valid...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"   Train size: {X_train.shape[0]}")
    print(f"   Valid size: {X_valid.shape[0]}")
    
    # 5. ОБУЧЕНИЕ МОДЕЛЕЙ
    print("\n" + "=" * 80)
    print("🤖 ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    print("\n📈 Обучение Logistic Regression...")
    lr_pipeline = train_logistic_regression(X_train, y_train)
    cross_validate_model(lr_pipeline, X_train, y_train)
    
    print("\n📈 Обучение KNN...")
    knn_pipeline = train_knn(X_train, y_train)
    cross_validate_model(knn_pipeline, X_train, y_train)
    
    print("\n📈 Обучение Decision Tree...")
    dt_pipeline = train_decision_tree(X_train, y_train)
    cross_validate_model(dt_pipeline, X_train, y_train)
    
    print("\n📈 Обучение Random Forest...")
    rf_pipeline = train_random_forest(X_train, y_train)
    cross_validate_model(rf_pipeline, X_train, y_train)
    
    print("\n📈 Обучение CatBoost...")
    catboost_pipeline = train_catboost(X_train, y_train)
    cross_validate_model(catboost_pipeline, X_train, y_train)
    
    print("\n📈 Обучение XGBoost...")
    xgb_pipeline = train_xgboost(X_train, y_train)
    cross_validate_model(xgb_pipeline, X_train, y_train)
    
    # 6. ОБУЧЕНИЕ DNN
    print("\n" + "=" * 80)
    print("🧠 ОБУЧЕНИЕ DNN")
    print("=" * 80)
    
    print("\n📈 Подготовка данных для DNN...")
    data_dict = prepare_dnn_data(X, y)
    
    print("\n📈 Обучение DNN...")
    dnn_model, preprocessor, dnn_test_acc = train_dnn(data_dict)
    
    # 7. СБОР ВСЕХ МОДЕЛЕЙ
    models = [
        lr_pipeline, knn_pipeline, dt_pipeline, rf_pipeline,
        catboost_pipeline, xgb_pipeline, dnn_model, preprocessor
    ]
    
    # 8. ВАЛИДАЦИЯ
    print("\n" + "=" * 80)
    print("📊 ВАЛИДАЦИЯ МОДЕЛЕЙ")
    print("=" * 80)
    
    print("\n📈 Оценка отдельных моделей...")
    proba_df, individual_results = evaluate_individual_models(
        models, X_valid, y_valid
    )
    
    print("\n📈 Оценка ансамблей...")
    ensemble_results, voting_clf, voting_hard, stacking_lr, stacking_ridge = \
        evaluate_ensembles(proba_df, models, X_train, y_train, X_valid, y_valid)
    
    # 9. РЕЗУЛЬТАТЫ ВАЛИДАЦИИ
    all_results = {**individual_results, **ensemble_results}
    
    results_df = pd.DataFrame(
        list(all_results.items()),
        columns=['Model', 'Validation Accuracy']
    ).sort_values('Validation Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    best_model, best_score = get_best_model(results_df)
    
    # 10. ГЕНЕРАЦИЯ SUBMISSION
    print("\n" + "=" * 80)
    print("📤 ГЕНЕРАЦИЯ SUBMISSION")
    print("=" * 80)
    
    print("\n📈 Создание submission для отдельных моделей...")
    single_submissions = generate_single_model_submissions(
        models, X_kaggle, test
    )
    
    print("\n📈 Создание ансамблей...")
    ensemble_predictions = create_all_ensembles(
        models, X_train, y_train, X_kaggle, test
    )
    
    # 11. ИТОГИ
    print("\n" + "=" * 80)
    print("✅ ПАЙПЛАЙН ЗАВЕРШЕН")
    print("=" * 80)
    print(f"\n🏆 Лучшая модель: {best_model}")
    print(f"📈 Validation Accuracy: {best_score:.4f}")
    print("\n📁 Созданы файлы в папке submissions/")


if __name__ == "__main__":
    main()
