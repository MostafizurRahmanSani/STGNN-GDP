from config import set_seed, PAST_WINDOW, HORIZON
from data.data_loader import get_st_datasets, unpack_graphs
from models.arima_model import train_arima, evaluate_arima_test_per_horizon
from models.gru_model import GRUModel, build_gru_dataset_from_stgnn, train_gru, evaluate_gru_test, unpack_graphs
from evaluation.visualization import plot_arima_model_selection, plot_gru_train_val_curves
import pickle
import torch
import pandas as pd
import numpy as np
from config import DOWNLOAD_PREFIX, FIRST_YEAR, LAST_YEAR, TRAIN_END, VAL_END

def get_arima_datasets():
    def load_all_gdp_series():
        gdp_dict = {}
        for file_year in range(FIRST_YEAR, LAST_YEAR + 1):
            df = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/Y_{file_year}.csv')
            target_year = file_year + 1
            col = str(target_year)
            if col not in df.columns:
                continue
            for _, row in df.iterrows():
                iso = row['iso_code']
                val = row[col]
                if pd.notna(val) and val > 0:
                    gdp_dict.setdefault(iso, []).append(
                        (target_year, np.log(val))
                    )
        for iso in gdp_dict:
            gdp_dict[iso] = sorted(gdp_dict[iso], key=lambda x: x[0])
        return gdp_dict

    def create_arima_sample(series, split_year):
        train = [v for y, v in series if y <= split_year]
        test = [v for y, v in series if y > split_year]
        if len(train) < 5 or len(test) < HORIZON:
            return None
        return np.array(train), np.array(test[:HORIZON])  

    gdp_dict = load_all_gdp_series()
    train, val, test = [], [], []

    for iso, series in gdp_dict.items():
        if len(series) < PAST_WINDOW + HORIZON:
            continue
        for year in range(FIRST_YEAR, LAST_YEAR - HORIZON + 1):
            split_year = year + PAST_WINDOW - 1
            sample = create_arima_sample(series, split_year)
            if sample is None:
                continue
            x_train, y_test = sample
            if split_year <= TRAIN_END:
                train.append((iso, x_train, y_test))
            elif split_year <= VAL_END:
                val.append((iso, x_train, y_test))
            else:
                test.append((iso, x_train, y_test))

    print(f"ARIMA → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

def main():
    print("=== ARIMA Baseline ===")
    set_seed(42)
    data_train, data_val, data_test = get_arima_datasets()

    arima_hyperparams = {
        "order_candidates": [(1,1,1), (2,1,1), (1,1,2)],
        "save_loss_interval": 1
    }

    print("Training ARIMA baseline...")
    arima_models, arima_loss_traj, best_order = train_arima(
        name_prefix="ARIMA_logGDP",
        hyperparams=arima_hyperparams,
        data_train=data_train,
        data_val=data_val
    )

    plot_arima_model_selection(arima_loss_traj)

    arima_test_results = evaluate_arima_test_per_horizon(
        arima_models,
        data_test,
        HORIZON
    )

    print("\nARIMA Test Results:")
    for k, v in arima_test_results.items():
        print(f"{k}: MSE={v['MSE']:.4f}, MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}, R²={v['R2']:.4f}")

    with open('arima_model.pkl', 'wb') as f:
        pickle.dump(arima_models, f)
    print("✓ ARIMA models saved to arima_model.pkl")

    print("\n=== GRU Model (Multivariate with 4 Features) ===")
    
    print("Loading STGNN datasets for GRU...")
    stgnn_train, stgnn_val, stgnn_test = get_st_datasets()
    
    print("Building multivariate datasets with 4 features...")
    X_train, Y_train, feature_scaler, target_scaler = build_gru_dataset_from_stgnn(
        stgnn_train, PAST_WINDOW, HORIZON, fit_scaler=True
    )
    X_val, Y_val = build_gru_dataset_from_stgnn(
        stgnn_val, PAST_WINDOW, HORIZON, feature_scaler, target_scaler, fit_scaler=False
    )
    X_test, Y_test = build_gru_dataset_from_stgnn(
        stgnn_test, PAST_WINDOW, HORIZON, feature_scaler, target_scaler, fit_scaler=False
    )

    gru_hyperparams = {
        "learning_rate": 0.001,
        "n_epochs": 2000,
        "save_loss_interval": 10,
        "print_interval": 50,
        "batch_size": 32,
        "patience": 50
    }

    set_seed(42)
    model = GRUModel(hidden_dim=32, horizon=HORIZON, input_dim=4)

    model, gru_losses, feature_scaler, target_scaler = train_gru(
        model,
        gru_hyperparams,
        X_train, Y_train,
        X_val, Y_val,
        feature_scaler,
        target_scaler
    )

    plot_gru_train_val_curves(gru_losses)

    gru_test_results, preds, truths = evaluate_gru_test(
        model,
        X_test, Y_test,
        target_scaler,
        HORIZON
    )

    print("\nGRU Test Results (Multivariate - 4 Features):")
    for k, v in gru_test_results.items():
        if k != "aggregate":
            print(f"{k}: MSE={v['MSE']:.4f}, MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}, R²={v['R2']:.4f}")
    print(f"\nAggregate: MAE={gru_test_results['aggregate']['MAE']:.4f}, RMSE={gru_test_results['aggregate']['RMSE']:.4f}, R²={gru_test_results['aggregate']['R2']:.4f}")


if __name__ == "__main__":
    main()