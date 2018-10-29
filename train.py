import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def run():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    train_x, train_y = load_data("train.csv")
    test_x, test_y = load_data("test.csv")

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, y_pred)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(model, "model")



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_data(path):
    data = pd.read_csv(path)
    x = data.drop(["quality"], axis=1)
    y = data[["quality"]]
    return x, y


if __name__ == "__main__":
    run()
