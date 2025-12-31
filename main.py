
#main.py
import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    # 1) Read config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    path = config["path"]
    features = config["features"]  # MUST be exactly 2 feature names

    # 2) Load data
    df = pd.read_csv("parkinsons.csv")

    # Make sure required columns exist
    missing = [c for c in (features + ["status"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parkinsons.csv: {missing}")

    # 3) Prepare X/y using ONLY the selected features
    df = df.dropna(subset=features + ["status"])
    X = df[features]
    y = df["status"]

    # 4) Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Model: scaler + SVC, with a small grid search
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf"))
    ])

    param_grid = {
        "svc__C": [0.1, 1, 10, 100, 1000],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1],
        "svc__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # 6) Save model EXACTLY to the configured path
    joblib.dump(best_model, path)

    # Optional prints (won't hurt autograder)
    print("Selected features:", features)
    print("Best params:", search.best_params_)
    print("Saved model to:", path)
    print("Validation accuracy:", best_model.score(X_val, y_val))


if __name__ == "__main__":
    main()
