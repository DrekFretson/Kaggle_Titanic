from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.config import NUM_COLS, CAT_COLS


def create_preprocessor():
    """Создание preprocessing pipeline"""
    # Pipeline для числовых признаков
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline для категориальных признаков
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Объединенный preprocessor
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_COLS),
        ('cat', cat_pipeline, CAT_COLS)
    ])
    
    return preprocessor
