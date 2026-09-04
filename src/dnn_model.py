import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, VALID_SIZE
from src.preprocessor import create_preprocessor


class MLP(nn.Module):
    """Многослойный персептрон"""
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)


def prepare_dnn_data(X, y):
    """Подготовка данных для DNN"""
    preprocessor = create_preprocessor()
    
    # Разделение на 3 части
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_dnn_train, X_dnn_val, y_dnn_train, y_dnn_val = train_test_split(
        X_train, y_train, test_size=VALID_SIZE,
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Preprocessing
    X_dnn_train_processed = preprocessor.fit_transform(X_dnn_train)
    X_dnn_val_processed = preprocessor.transform(X_dnn_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Sparse -> dense
    X_dnn_train_processed = X_dnn_train_processed.toarray()
    X_dnn_val_processed = X_dnn_val_processed.toarray()
    X_test_processed = X_test_processed.toarray()
    
    # В тензоры
    X_dnn_train_tensor = torch.tensor(X_dnn_train_processed, dtype=torch.float32)
    X_dnn_val_tensor = torch.tensor(X_dnn_val_processed, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    
    y_dnn_train_tensor = torch.tensor(y_dnn_train.values, dtype=torch.float32).view(-1, 1)
    y_dnn_val_tensor = torch.tensor(y_dnn_val.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    return {
        'X_train': X_dnn_train_tensor,
        'X_val': X_dnn_val_tensor,
        'X_test': X_test_tensor,
        'y_train': y_dnn_train_tensor,
        'y_val': y_dnn_val_tensor,
        'y_test': y_test_tensor,
        'preprocessor': preprocessor
    }


def train_dnn(data_dict, epochs=100, patience=10, lr=0.001, batch_size=32):
    """Обучение DNN"""
    # DataLoader
    train_dataset = TensorDataset(data_dict['X_train'], data_dict['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Модель
    input_size = data_dict['X_train'].shape[1]
    model = MLP(input_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_outputs = model(data_dict['X_val'])
            valid_loss = criterion(valid_outputs, data_dict['y_val'])
            valid_pred = (torch.sigmoid(valid_outputs) >= 0.5).float()
            valid_accuracy = (valid_pred == data_dict['y_val']).float().mean()
        
        # Early stopping check
        if valid_loss.item() < best_valid_loss:
            best_valid_loss = valid_loss.item()
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:3d} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Valid loss: {valid_loss.item():.4f} | '
                  f'Valid accuracy: {valid_accuracy.item():.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Test
    model.eval()
    with torch.no_grad():
        test_outputs = model(data_dict['X_test'])
        test_loss = criterion(test_outputs, data_dict['y_test'])
        test_pred = (torch.sigmoid(test_outputs) >= 0.5).float()
        test_accuracy = (test_pred == data_dict['y_test']).float().mean()
    
    print(f'Final test accuracy: {test_accuracy.item():.4f}')
    
    return model, data_dict['preprocessor'], test_accuracy.item()


def mlp_predict(model, X, preprocessor):
    """Предсказание с помощью DNN"""
    X_processed = preprocessor.transform(X)
    X_processed = X_processed.toarray()
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (torch.sigmoid(outputs) >= 0.5).int().numpy().ravel()
    
    return predictions
