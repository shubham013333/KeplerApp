import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_kepler


def train_and_save(csv_path: str, out_path: str = "model_pipeline.pkl"):
    df = load_kepler(csv_path)
    features = [
        "koi_period", "koi_time0bk", "koi_impact", "koi_duration",
        "koi_depth", "koi_prad", "koi_teq", "koi_srad", "koi_smass",
        "koi_steff", "koi_model_snr", "koi_dor", "koi_insol",
        "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"
    ]

    df = df.dropna(subset=features + ["koi_disposition"])

    X = df[features]
    y = df["koi_disposition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("clf", clf)])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    folder = os.path.dirname(out_path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    joblib.dump(pipe, out_path)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Pipeline saved successfully at {out_path} ({os.path.getsize(out_path)} bytes)")
    else:
        print("Warning: Pipeline file not saved correctly!")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../Dataset/cumulative_2025.09.27_01.08.12.csv"
    train_and_save(csv_path)
