import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_arima(
    name_prefix,
    hyperparams,
    data_train,
    data_val
):
    print(f"--- Starting ARIMA Training for '{name_prefix}' ---")

    best_mae = float("inf")
    best_order = None
    losses = []

    for order in hyperparams["order_candidates"]:
        val_mae = []
        val_mse = []
        val_r2 = []

        for iso, x_train, y_true in data_val:
            try:
                model = ARIMA(x_train, order=order)
                fitted = model.fit()
                y_pred = fitted.forecast(steps=len(y_true))

                val_mae.append(mean_absolute_error(y_true, y_pred))
                val_mse.append(mean_squared_error(y_true, y_pred))
                val_r2.append(r2_score(y_true, y_pred))
            except:
                continue

        if len(val_mae) == 0:
            continue

        avg_mae = np.mean(val_mae)
        avg_mse = np.mean(val_mse)
        avg_r2 = np.mean(val_r2)

        losses.append((order, avg_mse, avg_mae, avg_r2))

        print(
            f"Order {order} | "
            f"Val MSE {avg_mse:.4f} | "
            f"Val MAE {avg_mae:.4f} | "
            f"R² {avg_r2:.4f}"
        )

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_order = order

    print(f"\n Best ARIMA order: {best_order} (Val MAE = {best_mae:.4f})")

    trained_models = {}

    for iso, x_train, _ in data_train:
        try:
            model = ARIMA(x_train, order=best_order)
            trained_models[iso] = model.fit()
        except:
            continue

    print(f"--- Finished ARIMA Training for '{name_prefix}' ---")
    return trained_models, losses, best_order

def evaluate_arima_test_per_horizon(
    arima_models,
    data_test,
    horizon
):
    mse = [[] for _ in range(horizon)]
    mae = [[] for _ in range(horizon)]
    rmse = [[] for _ in range(horizon)]
    y_true_all = [[] for _ in range(horizon)]
    y_pred_all = [[] for _ in range(horizon)]

    for iso, x_train, y_true in data_test:
        if iso not in arima_models:
            continue

        try:
            model = arima_models[iso]
            y_pred = model.forecast(steps=horizon)
        except:
            continue

        effective_h = min(horizon, len(y_true), len(y_pred))

        for h in range(effective_h):
            yt = y_true[h]
            yp = y_pred[h]

            mse[h].append((yt - yp) ** 2)
            mae[h].append(abs(yt - yp))
            rmse[h].append((yt - yp) ** 2)

            y_true_all[h].append(yt)
            y_pred_all[h].append(yp)

    results = {}
    for h in range(horizon):
        if len(y_true_all[h]) < 2:
            continue

        results[f"t+{h+1}"] = {
            "MSE": np.mean(mse[h]),
            "MAE": np.mean(mae[h]),
            "RMSE": np.sqrt(np.mean(rmse[h])),
            "R2": r2_score(y_true_all[h], y_pred_all[h])
        }

    return results