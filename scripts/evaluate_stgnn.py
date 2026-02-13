import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import set_seed
from data.data_loader import get_st_datasets
from models.stgnn import STGNN
from evaluation.metrics import regression_metrics
from evaluation.visualization import plot_predictions_vs_truth

def main():
    print("Loading STGNN datasets...")
    _, _, data_test = get_st_datasets()

    print("Loading saved STGNN model...")
    set_seed(42)
    model = STGNN(
        in_dim=4,
        hidden_dim=32,
        agg="mean",
        comb="gru",
        norm=True
    )

 
    model.load_state_dict(torch.load("stgnn_current.pt", map_location='cpu', weights_only=True))
    model.eval()
    print("Model loaded successfully!")

    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    

    metrics = regression_metrics(model, data_test)
    
    print("\n" + "="*60)
    print("GENERATING PREDICTION PLOTS")
    print("="*60)
    

    plot_predictions_vs_truth(model, data_test, horizon=0)
    plot_predictions_vs_truth(model, data_test, horizon=1)
    plot_predictions_vs_truth(model, data_test, horizon=2)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()