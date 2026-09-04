import numpy as np
import pandas as pd


def extract_ticket_prefix(ticket):
    """Извлечение префикса из номера билета"""
    prefix = ''.join(char for char in ticket if not char.isdigit())
    prefix = prefix.replace(' ', '').replace('.', '').replace('/', '')
    return prefix if prefix else 'NONE'


def age_group(age):
    """Группировка возраста"""
    if pd.isna(age):
        return 'Unknown'
    elif age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 35:
        return 'YoungAdult'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'


def add_features(df, train_df=None):
    """Добавление всех новых признаков"""
    df = df.copy()
    
    # Размер семьи
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Путешествует один
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title из Name
    df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.')[0]
    
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don',
                   'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    
    title_mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
    df['Title'] = df['Title'].replace(title_mapping)
    
    # Размер группы по билету
    if train_df is not None:
        ticket_counts = train_df['Ticket'].value_counts()
    else:
        ticket_counts = df['Ticket'].value_counts()
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
    
    # Стоимость билета на человека
    df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
    
    # Палуба
    df['Deck'] = df['Cabin'].str[0]
    
    # Наличие каюты
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    # Возрастная группа
    df['AgeGroup'] = df['Age'].apply(age_group)
    
    # Группа стоимости билета
    if train_df is not None:
        fare_bins = train_df['Fare'].quantile([0, 0.25, 0.5, 0.75, 1]).values
        fare_bins = np.unique(fare_bins)
    else:
        fare_bins = df['Fare'].quantile([0, 0.25, 0.5, 0.75, 1]).values
        fare_bins = np.unique(fare_bins)
    
    labels = ['Low', 'Medium', 'High', 'VeryHigh'][:len(fare_bins) - 1]
    df['FareGroup'] = pd.cut(
        df['Fare'],
        bins=fare_bins,
        labels=labels,
        include_lowest=True
    )
    df['FareGroup'] = df['FareGroup'].astype(object).fillna('Unknown')
    
    # Префикс билета
    df['TicketPrefix'] = df['Ticket'].apply(extract_ticket_prefix)
    
    # Взаимодействия
    df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
    df['Sex_AgeGroup'] = df['Sex'] + '_' + df['AgeGroup']
    
    return df


def engineer_features(train, test):
    """Создание признаков для train и test"""
    train = add_features(train, train_df=train)
    test = add_features(test, train_df=train)
    return train, test
