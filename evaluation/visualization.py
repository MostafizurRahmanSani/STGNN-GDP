import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from data.data_loader import unpack_graphs

def plot_stgnn_training_curves(losses):
    epochs = [e for e, _, _, _, _ in losses]
    train_mse = [tmse for _, tmse, _, _, _ in losses]
    val_mse = [vmse for _, _, _, vmse, _ in losses]
    train_mae = [tmae for _, _, tmae, _, _ in losses]
    val_mae = [vmae for _, _, _, _, vmae in losses]

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_mse, label='Train MSE', alpha=0.8)
    plt.plot(epochs, val_mse, label='Val MSE', alpha=0.8)
    plt.title('STGNN Training Progress - MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mae, label='Train MAE', alpha=0.8)
    plt.plot(epochs, val_mae, label='Val MAE', alpha=0.8)
    plt.title('STGNN Training Progress - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    best_epoch = epochs[val_mae.index(min(val_mae))]
    print(f"\nBest Validation MAE: {min(val_mae):.4f} at epoch {best_epoch}")
    print(f"Best Validation MSE: {min(val_mse):.4f} at epoch {epochs[val_mse.index(min(val_mse))]}")



def plot_predictions_vs_truth(model, dataset, horizon=0):
    model.eval()
    device = next(model.parameters()).device

    ground_truth = []
    preds = []

    with torch.no_grad():
        for graphs, y in dataset:
            xs, edge_indices, edge_attrs = unpack_graphs(graphs)

            xs = xs.to(device)
            edge_indices = [ei.to(device) for ei in edge_indices]
            edge_attrs = [ea.to(device) for ea in edge_attrs]
            y = y.to(device)

            out = model(xs, edge_indices, edge_attrs)

            ground_truth.append(y[:, horizon].cpu().numpy())
            preds.append(out[:, horizon].cpu().numpy())

    ground_truth = np.concatenate(ground_truth)
    preds = np.concatenate(preds)

    max_x = ground_truth.max() * 1.15

    plt.figure(figsize=(6, 6))
    plt.scatter(ground_truth, preds, s=5, alpha=0.6)
    plt.plot([0, max_x], [0, max_x], 'r--', linewidth=1)
    plt.xlabel("Actual log GDP")
    plt.ylabel("Predicted log GDP")
    plt.title(f"STGNN Predictions vs Ground Truth (t+{horizon+1})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_subgraph_heatmap(model, data, translator, country_list):
    name_to_code = {v: k for k, v in translator.code_to_name.items()}
    code_to_node_id = {code: i for i, code in enumerate(data.iso_codes)}

    node_ids = []
    country_names = []

    for name in country_list:
        if name in name_to_code:
            code = name_to_code[name]
            if code in code_to_node_id:
                node_ids.append(code_to_node_id[code])
                country_names.append(name)
            else:
                print(f"Warning: '{name}' not found in the graph for this year.")
        else:
            print(f"Warning: '{name}' not found in translator.")

    if len(node_ids) == 0:
        print("No valid countries to plot.")
        return

    node_ids = torch.tensor(node_ids, dtype=torch.long)
    sub = data.subgraph(node_ids)

    edge_index = sub.edge_index.cpu().numpy()
    edge_weights = sub.edge_attr.mean(dim=1).cpu().numpy()

    n = sub.num_nodes
    influence = np.zeros((n, n))
    influence[edge_index[1], edge_index[0]] = edge_weights

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        influence,
        xticklabels=country_names,
        yticklabels=country_names,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f"
    )
    plt.title("Trade Edge Influence Heatmap")
    plt.xlabel("Target Country")
    plt.ylabel("Source Country")
    plt.tight_layout()
    plt.show()

def plot_gru_train_val_curves(losses):
    epochs = [l[0] for l in losses]

    train_mse = [l[1] for l in losses]
    train_mae = [l[2] for l in losses]
    val_mse = [l[3] for l in losses]
    val_mae = [l[4] for l in losses]

    plt.figure()
    plt.plot(epochs, train_mse, label="Train MSE")
    plt.plot(epochs, val_mse, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("GRU Train vs Validation MSE")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_mae, label="Train MAE")
    plt.plot(epochs, val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("GRU Train vs Validation MAE")
    plt.legend()
    plt.show()

def plot_arima_model_selection(arima_losses):
    orders = [str(l[0]) for l in arima_losses]
    val_mse = [l[1] for l in arima_losses]
    val_mae = [l[2] for l in arima_losses]

    plt.figure()
    plt.plot(orders, val_mse, marker='o', label="Val MSE")
    plt.xlabel("ARIMA Order")
    plt.ylabel("MSE")
    plt.title("ARIMA Validation MSE by Order")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(orders, val_mae, marker='o', label="Val MAE")
    plt.xlabel("ARIMA Order")
    plt.ylabel("MAE")
    plt.title("ARIMA Validation MAE by Order")
    plt.legend()
    plt.show()


