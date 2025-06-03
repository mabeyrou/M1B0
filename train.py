import mlflow
from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from models.models import train_model, model_predict
import pandas as pd
from os.path import join as join
import joblib

loaded_model = mlflow.sklearn.load_model("runs:/ea761799bed94e7fbb3ad8fe3be38cab/model")

mlflow.set_experiment("M1B0")

with mlflow.start_run(run_name="new_data_two_trainings_more_epochs"):
    df_path = 'df_new.csv'
    df = pd.read_csv(join('data', df_path))

    X, y, _ = preprocessing(df)

    X_train, X_test, y_train, y_test = split(X, y)

    retrained_model, hist = train_model(loaded_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=300)
    y_pred = model_predict(loaded_model, X_test)
    perf = evaluate_performance(y_test, y_pred)  

    mlflow.log_metric("MSE", perf['MSE'])
    mlflow.log_metric("MAE", perf['MAE'])
    mlflow.log_metric("R²", perf['R²'])
    mlflow.sklearn.log_model(loaded_model, "model")
    joblib.dump(loaded_model, join('models','model_2025_06_03.pkl'))