import os
from pathlib import Path

# Пути к данным
DATA_DIR = Path('data')
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'

# Пути для сохранения
MODELS_DIR = Path('models')
OUTPUTS_DIR = Path('outputs')
SUBMISSIONS_DIR = Path('submissions')

# Создаем директории
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Параметры обучения
RANDOM_STATE = 0
TEST_SIZE = 0.2
VALID_SIZE = 0.2

# Числовые признаки
NUM_COLS = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'TicketGroupSize',
    'FarePerPerson', 'HasCabin'
]

# Категориальные признаки
CAT_COLS = [
    'Sex', 'Embarked', 'Title', 'Deck',
    'AgeGroup', 'FareGroup', 'TicketPrefix',
    'Sex_Pclass', 'Sex_AgeGroup'
]
