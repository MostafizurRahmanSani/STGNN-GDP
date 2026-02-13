import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import pandas as pd
import numpy as np
from config import set_seed, PAST_WINDOW, HORIZON, DOWNLOAD_PREFIX, FIRST_YEAR, LAST_YEAR, TRAIN_END, VAL_END
from models.gru_model import GRUModel, build_gru_dataset_from_stgnn, evaluate_gru_test
from models.arima_model import evaluate_arima_test_per_horizon
from evaluation.visualization import plot_gru_train_val_curves
from data.data_loader import get_st_datasets, unpack_graphs

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
        if len(train) < 5 or len(test) == 0:
            return None
        return np.array(train), np.array(test)

    gdp_dict = load_all_gdp_series()
    test = []

    for iso, series in gdp_dict.items():
        if len(series) < PAST_WINDOW + HORIZON:
            continue
        for year in range(FIRST_YEAR, LAST_YEAR - HORIZON + 1):
            split_year = year + PAST_WINDOW - 1
            if split_year > VAL_END: 
                sample = create_arima_sample(series, split_year)
                if sample is not None:
                    x_train, y_test = sample
                    test.append((iso, x_train, y_test))

    print(f"ARIMA Test samples: {len(test)}")
    return [], [], test  

def main():
    print("="*60)
    print("EVALUATING SAVED BASELINE MODELS")
    print("="*60)
    
    print("\n ARIMA MODEL")
    print("-" * 40)

    _, _, data_test = get_arima_datasets()
    
    try:
        with open('arima_model.pkl', 'rb') as f:
            arima_models = pickle.load(f)
        print("ARIMA models loaded from arima_model.pkl")
        
        arima_results = evaluate_arima_test_per_horizon(
            arima_models, data_test, HORIZON
        )
        
        print("\nARIMA Test Results:")
        for k, v in arima_results.items():
            print(f"  {k}: MSE={v['MSE']:.4f}, MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}, R²={v['R2']:.4f}")
            
    except FileNotFoundError:
        print("ARIMA models not found. Train first with: python scripts/run_arima_gru.py")
    
    print("\nGRU MODEL")
    print("-" * 40)
    
    try:
        _, _, stgnn_test = get_st_datasets()
        
        checkpoint = torch.load("gru_model.pt", map_location='cpu')
        feature_scaler = checkpoint['feature_scaler']
        target_scaler = checkpoint['target_scaler']
        
        X_test, Y_test = build_gru_dataset_from_stgnn(
            stgnn_test, PAST_WINDOW, HORIZON, 
            feature_scaler, target_scaler, fit_scaler=False
        )

        set_seed(42)
        model = GRUModel(hidden_dim=32, horizon=HORIZON, input_dim=4)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("GRU model loaded from gru_model.pt")

        gru_results, _, _ = evaluate_gru_test(
            model, X_test, Y_test, target_scaler, HORIZON
        )
        
        print("\nGRU Test Results (Multivariate - 4 Features):")
        for k, v in gru_results.items():
            if k != "aggregate":
                print(f"  {k}: MSE={v['MSE']:.4f}, MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}, R²={v['R2']:.4f}")
            
    except FileNotFoundError:
        print("GRU model not found. Train first with: python scripts/run_arima_gru.py")
    except Exception as e:
        print(f"Error loading GRU model: {e}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()