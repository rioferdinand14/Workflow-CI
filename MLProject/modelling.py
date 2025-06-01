import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def main():
    # Atur MLflow ke penyimpanan lokal di folder mlruns/
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("BreastCancer_Classification")

    # Load data preprocessing
    X_train = pd.read_csv("breast_cancer_preprocessing/X_train.csv")
    X_test  = pd.read_csv("breast_cancer_preprocessing/X_test.csv")
    y_train = pd.read_csv("breast_cancer_preprocessing/y_train.csv").values.ravel()
    y_test  = pd.read_csv("breast_cancer_preprocessing/y_test.csv").values.ravel()

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    # Mulai run MLflow
    with mlflow.start_run():
        # Contoh model: Random Forest
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        try:
            # Jika classifier mendukung predict_proba
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        except Exception:
            print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
