import pandas as pd
from src.config import TRAIN_PATH, TEST_PATH


def load_data():
    """Загрузка исходных данных"""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def get_features_target(train):
    """Разделение на признаки и целевую переменную"""
    X = train.drop(columns='Survived')
    y = train['Survived']
    return X, y
