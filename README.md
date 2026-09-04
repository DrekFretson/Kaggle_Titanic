# Kaggle_Titanic

Предсказание выживаемости пассажиров Титаника (Kaggle competition).

## Структура проекта

- `src/` – исходный код
- `notebooks/eda.ipynb` – исследовательский анализ данных (EDA)
- `submissions/` – сгенерированные файлы для отправки на Kaggle
- `outputs/` – результаты обучения и графики
- `README.md` – описание проекта
- `requirements.txt` – зависимости проекта

## Используемые модели

1. **Логистическая регрессия** (базовый классификатор)
2. **KNN** (метод ближайших соседей)
3. **Decision Tree** (решающее дерево)
4. **Random Forest** (случайный лес)
5. **CatBoost** (градиентный бустинг)
6. **XGBoost** (градиентный бустинг)
7. **DNN (PyTorch)** – полносвязная нейронная сеть с BatchNorm и Dropout

## Feature Engineering

- **FamilySize** – размер семьи (`SibSp + Parch + 1`)
- **IsAlone** – путешествует ли пассажир один
- **Title** – обращение, извлечённое из имени
- **TicketGroupSize** – количество пассажиров с одинаковым билетом
- **FarePerPerson** – стоимость билета на одного человека
- **Deck** – палуба, извлечённая из номера каюты
- **HasCabin** – наличие информации о каюте
- **AgeGroup** – возрастная группа
- **FareGroup** – категория стоимости билета
- **TicketPrefix** – префикс билета
- **Sex_Pclass** – комбинация пола и класса
- **Sex_AgeGroup** – комбинация пола и возрастной группы

## Ансамбли

- **Simple Average** – усреднение вероятностей всех моделей
- **Weighted Average** – взвешенное усреднение вероятностей
- **Soft Voting** – голосование по вероятностям
- **Hard Voting** – голосование по классам
- **Stacking + Logistic Regression** – stacking с логистической регрессией
- **Stacking + Ridge Classifier** – stacking с Ridge Classifier

## Результаты (на валидации)

| **Модель / Ансамбль** | **Validation Accuracy** |
| --------------------- | ----------------------: |
| **Soft Voting**       | **0.8715** |
| **Hard Voting**       | 0.8659 |
| **Stacking Ridge**    | 0.8603 |
| DNN                   | 0.8547 |
| Stacking LR           | 0.8547 |
| Decision Tree         | 0.8268 |
| Simple Average        | 0.8268 |
| Weighted Average      | 0.8268 |
| Random Forest         | 0.8212 |
| Logistic Regression   | 0.8156 |
| CatBoost              | 0.8156 |
| KNN                   | 0.8101 |
| XGBoost               | 0.8101 |

## Запуск проекта

1. Установка зависимостей:

```bash
pip install -r requirements.txt
```
2. Запуск проекта:
```bash
python main.py
```
