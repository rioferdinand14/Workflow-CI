import pandas as pd
import numpy as np
import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

def main():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rfe = RFE(estimator=rf_selector, n_features_to_select=10)
    rfe.fit(df, y)
    
    selected_features = df.columns[rfe.get_support()].tolist()
    
    print("TOP 10 FEATURES:")
    print(selected_features)

    X_selected = df[selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("file:./modelling/mlruns")
    mlflow.set_experiment("BreastCancer")
    
    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")

        mlflow.log_param("selected_features", str(selected_features))
        mlflow.log_metric("final_accuracy", acc)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        production_path = "./monitoring/model_production" 
        
        if os.path.exists(production_path):
            shutil.rmtree(production_path)
        
        # mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run.info.run_id}/model", dst_path=production_path)

if __name__ == "__main__":
    main()