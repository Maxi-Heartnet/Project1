import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ['bedrooms', 'area_m2']
CATEGORICAL_FEATURES = ['sector', 'property_type']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PRICE_MIN = 10_000
PRICE_MAX = 5_000_000


def clean(df):
    df = df.dropna(subset=['price', 'area_m2'])
    df = df[(df['price'] >= PRICE_MIN) & (df['price'] <= PRICE_MAX)].copy()
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
    return df


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', 'passthrough', NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
    ])


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df = clean(df)
    X = df[ALL_FEATURES]
    y = df['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    return (
        preprocessor.transform(X_train),
        preprocessor.transform(X_test),
        y_train,
        y_test,
        preprocessor,
    )
