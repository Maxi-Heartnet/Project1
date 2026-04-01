import sys
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from ml.prepare import load_and_prepare

MODEL_PATH = 'ml/model.pkl'


def train(csv_path, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(csv_path)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
    print(f"R²:  {r2_score(y_test, y_pred):.3f}")
    joblib.dump({"model": model, "encoder": preprocessor}, model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else 'data/listings.csv'
    train(csv)
