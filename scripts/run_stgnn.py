import torch
from config import set_seed, PAST_WINDOW, HORIZON
from data.data_loader import get_st_datasets, unpack_graphs
from models.stgnn import STGNN
from training.train_stgnn import train_st
from evaluation.metrics import regression_metrics
from evaluation.visualization import plot_predictions_vs_truth, plot_stgnn_training_curves

def main():
    print("Loading STGNN datasets...")
    data_train, data_val, data_test = get_st_datasets()

    hyperparams = {
        'batch_size': 4,
        'save_loss_interval': 10,
        'print_interval': 50,
        'save_model_interval': 250,
        'n_epochs': 2000,
        'learning_rate': 0.001
    }

    set_seed(42)

    print("Training STGNN model...")
    
    model = STGNN(
        in_dim=4,
        hidden_dim=32,
        agg="mean",
        comb="gru",
        norm=True
    )

    model, model_loss_traj = train_st(
        model,
        name_prefix="STGNN_dynamic",
        hyperparams=hyperparams,
        data_train=data_train,
        data_val=data_val
    )

    torch.save(model.state_dict(), "stgnn_current.pt")
    print("Model saved to stgnn_current.pt")

    print("\nPlotting training curves:")
    plot_stgnn_training_curves(model_loss_traj)

    print("\nEvaluating on training set:")
    regression_metrics(model, data_train)

    print("\nEvaluating on validation set:")
    regression_metrics(model, data_val)

    print("\nEvaluating on test set:")
    regression_metrics(model, data_test)

    print("\nGenerating prediction plots...")
    plot_predictions_vs_truth(model, data_test, horizon=0)
    plot_predictions_vs_truth(model, data_test, horizon=1)
    plot_predictions_vs_truth(model, data_test, horizon=2)

if __name__ == "__main__":
    main()