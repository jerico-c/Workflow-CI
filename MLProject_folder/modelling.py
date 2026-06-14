import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os

# Set tracking URI ke folder lokal khusus untuk CI Pipeline
mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
mlflow.set_experiment("CI_Credit_Risk")

with mlflow.start_run() as run:
    # 1. Load Data
    X_train = pd.read_csv('dataset_preprocessing/X_train_clean.csv')
    y_train = pd.read_csv('dataset_preprocessing/y_train_clean.csv').squeeze()

    # 2. Latih Model (Versi cepat untuk CI)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # 3. Log Model
    mlflow.sklearn.log_model(model, "model")

    # 4. Simpan Run ID ke file teks agar bisa dibaca oleh GitHub Actions
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    print(f" Model CI berhasil dilatih dengan Run ID: {run_id}")