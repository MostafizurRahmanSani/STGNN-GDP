import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.data_loader import unpack_graphs

def regression_metrics(model, dataset):
    model.eval()
    device = next(model.parameters()).device

    y_true = {0: [], 1: [], 2: []}
    y_pred = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for graphs, y in dataset:
            xs, edge_indices, edge_attrs = unpack_graphs(graphs)

            xs = xs.to(device)
            edge_indices = [ei.to(device) for ei in edge_indices]
            edge_attrs = [ea.to(device) for ea in edge_attrs]
            y = y.to(device)

            pred = model(xs, edge_indices, edge_attrs)

            for h in range(3):
                y_true[h].append(y[:, h].cpu().numpy())
                y_pred[h].append(pred[:, h].cpu().numpy())

    results = {}

    print("\nPer-Horizon Regression Metrics\n")

    for h in range(3):
        yt = np.concatenate(y_true[h])
        yp = np.concatenate(y_pred[h])

        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(yt, yp)
        r2 = r2_score(yt, yp)

        eps = 1e-8
        accuracy = np.mean(
            np.abs(yt - yp) / (np.abs(yt) + eps) < 0.10
        ) * 100

        print(
            f"Horizon t+{h+1}: "
            f"MAE={mae:.6f} | "
            f"MSE={mse:.6f} | "
            f"RMSE={rmse:.6f} | "
            f"R²={r2:.6f} | "
            f"Acc(<10%)={accuracy:.2f}%"
        )

        results[f"t+{h+1}"] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Accuracy": accuracy
        }

    return results

def evaluate_arima_test_per_horizon(arima_models, data_test, horizon):
    from models.arima_model import evaluate_arima_test_per_horizon as eval_core
    return eval_core(arima_models, data_test, horizon)

def evaluate_gru_test(model, X_test, Y_test, horizon):
    from models.gru_model import evaluate_gru_test as eval_core
    return eval_core(model, X_test, Y_test, horizon)