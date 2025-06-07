import mlflow
from mlflow.models import infer_signature
from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from models.models import train_model, model_predict, create_nn_model
import pandas as pd
from os.path import join as join
import joblib

df_old = pd.read_csv(join('data','df_old.csv'))
df_new = pd.read_csv(join('data','df_new.csv'))

old_model = joblib.load(join('models','model_2024_08.pkl'))

mlflow.set_experiment("Retraining loan prediction model")
mlflow.autolog()

# with mlflow.start_run():
#     retraining_data = df_old
#     X, y, _ = preprocessing(retraining_data)

#     X_train, X_test, y_train, y_test = split(X, y)

#     retrained_model, _ = train_model(old_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)
#     y_pred = model_predict(old_model, X_test)
#     perf = evaluate_performance(y_test, y_pred)  

#     mlflow.log_metric("R²", perf['R²'])

#     mlflow.set_tag("Training Info", "Old model retrained with old data")

#     signature = infer_signature(X_train, model_predict(retrained_model, X_train))

#     model_info = mlflow.sklearn.log_model(
#         sk_model=retrained_model,
#         artifact_path="iris_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="loan-prediction",
#     )

# with mlflow.start_run():
#     retraining_data = df_new
#     X, y, _ = preprocessing(retraining_data)

#     X_train, X_test, y_train, y_test = split(X, y)

#     retrained_model, _ = train_model(old_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)
#     y_pred = model_predict(old_model, X_test)
#     perf = evaluate_performance(y_test, y_pred)  

#     mlflow.log_metric("R²", perf['R²'])

#     mlflow.set_tag("Training Info", "Old model retrained with new data")

#     signature = infer_signature(X_train, model_predict(retrained_model, X_train))

#     model_info = mlflow.sklearn.log_model(
#         sk_model=retrained_model,
#         artifact_path="iris_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="loan-prediction",
#     )

# with mlflow.start_run():
#     retraining_data = df_new
#     X, y, _ = preprocessing(retraining_data)

#     X_train, X_test, y_train, y_test = split(X, y)

#     retrained_model, _ = train_model(old_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=300)
#     y_pred = model_predict(old_model, X_test)
#     perf = evaluate_performance(y_test, y_pred)  

#     mlflow.log_metric("R²", perf['R²'])

#     mlflow.set_tag("Training Info", "Old model retrained with new data and 300 epochs")

#     signature = infer_signature(X_train, model_predict(retrained_model, X_train))

#     model_info = mlflow.sklearn.log_model(
#         sk_model=retrained_model,
#         artifact_path="iris_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="loan-prediction",
#     )

# with mlflow.start_run():
#     training_data = df_new
#     X, y, _ = preprocessing(training_data)
#     X_train, X_test, y_train, y_test = split(X, y)
#     new_model = create_nn_model(X_train.shape[1])
#     new_model, _ = train_model(new_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=100)
#     joblib.dump(new_model, join('models','model_2025_06.pkl'))
#     y_pred = model_predict(new_model, X_test)
#     perf = evaluate_performance(y_test, y_pred)  

#     mlflow.log_metric("R²", perf['R²'])

#     mlflow.set_tag("Training Info", "New model, trained with new data only once (50 epochs)")

#     signature = infer_signature(X_train, model_predict(new_model, X_train))

#     model_info = mlflow.sklearn.log_model(
#     sk_model=new_model,
#     artifact_path="iris_model",
#     signature=signature,
#     input_example=X_train,
#     registered_model_name="loan-prediction",
#    )

new_model = joblib.load(join('models','model_2025_06.pkl'))

# with mlflow.start_run():
#     retraining_data = df_new
#     X, y, _ = preprocessing(retraining_data)

#     X_train, X_test, y_train, y_test = split(X, y)

#     retrained_model, _ = train_model(new_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)
#     y_pred = model_predict(new_model, X_test)
#     perf = evaluate_performance(y_test, y_pred)  

#     mlflow.log_metric("R²", perf['R²'])

#     mlflow.set_tag("Training Info", "New model (trained with new data) retrained with new data")

#     signature = infer_signature(X_train, model_predict(retrained_model, X_train))

#     model_info = mlflow.sklearn.log_model(
#         sk_model=retrained_model,
#         artifact_path="iris_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="loan-prediction",
#     )

# with mlflow.start_run():
#     retraining_data = df_old
#     X, y, _ = preprocessing(retraining_data)

#     X_train, X_test, y_train, y_test = split(X, y)

#     retrained_model, _ = train_model(new_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)
#     y_pred = model_predict(new_model, X_test)
#     perf = evaluate_performance(y_test, y_pred)  

#     mlflow.log_metric("R²", perf['R²'])

#     mlflow.set_tag("Training Info", "New model (trained with new data) retrained with old data")

#     signature = infer_signature(X_train, model_predict(retrained_model, X_train))

#     model_info = mlflow.sklearn.log_model(
#         sk_model=retrained_model,
#         artifact_path="iris_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="loan-prediction",
#     )

with mlflow.start_run():
    retraining_data = df_new
    X, y, _ = preprocessing(retraining_data)

    X_train, X_test, y_train, y_test = split(X, y)

    retrained_model, _ = train_model(new_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=300)
    y_pred = model_predict(new_model, X_test)
    perf = evaluate_performance(y_test, y_pred)  

    mlflow.log_metric("R²", perf['R²'])

    mlflow.set_tag("Training Info", "New model (trained with new data) retrained with new data, 300 epochs")

    signature = infer_signature(X_train, model_predict(retrained_model, X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=retrained_model,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="loan-prediction",
    )

# with mlflow.start_run():
#     retraining_data = df_new
#     X, y, _ = preprocessing(retraining_data)

#     X_train, X_test, y_train, y_test = split(X, y)

#     retrained_model, _ = train_model(new_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)
#     retrained_model, _ = train_model(new_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)
#     retrained_model, _ = train_model(new_model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)
#     y_pred = model_predict(new_model, X_test)
#     perf = evaluate_performance(y_test, y_pred)  

#     mlflow.log_metric("R²", perf['R²'])

#     mlflow.set_tag("Training Info", "New model (trained with new data) retrained with new data 3 times")

#     signature = infer_signature(X_train, model_predict(retrained_model, X_train))

#     model_info = mlflow.sklearn.log_model(
#         sk_model=retrained_model,
#         artifact_path="iris_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="loan-prediction",
#     )