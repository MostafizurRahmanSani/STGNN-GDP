import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
from data.data_loader import unpack_graphs
from config import set_seed

def train_st(model, name_prefix, hyperparams, data_train, data_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting ST Training for '{name_prefix}' on {device} ---")

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"]
    )

    best_mae = float("inf")
    best_state = None
    losses = []

    for epoch in range(hyperparams["n_epochs"]):
        model.train()
        train_mse = 0.0
        train_mae = 0.0

        for graphs, y in data_train:
            xs, edge_indices, edge_attrs = unpack_graphs(graphs)

            xs = xs.to(device)
            edge_indices = [ei.to(device) for ei in edge_indices]
            edge_attrs = [ea.to(device) for ea in edge_attrs]
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(xs, edge_indices, edge_attrs)
            loss = F.mse_loss(pred, y)

            loss.backward()
            optimizer.step()

            train_mse += loss.item()
            train_mae += F.l1_loss(pred, y).item()

        train_mse /= len(data_train)
        train_mae /= len(data_train)

        if epoch % hyperparams["save_loss_interval"] == 0:
            model.eval()
            val_mse = 0.0
            val_mae = 0.0
            r2s = []

            with torch.no_grad():
                for graphs, y in data_val:
                    xs, edge_indices, edge_attrs = unpack_graphs(graphs)

                    xs = xs.to(device)
                    edge_indices = [ei.to(device) for ei in edge_indices]
                    edge_attrs = [ea.to(device) for ea in edge_attrs]
                    y = y.to(device)

                    pred = model(xs, edge_indices, edge_attrs)

                    val_mse += F.mse_loss(pred, y).item()
                    val_mae += F.l1_loss(pred, y).item()

                    r2s.append(
                        r2_score(
                            y.cpu().numpy().reshape(-1),
                            pred.cpu().numpy().reshape(-1)
                        )
                    )

            val_mse /= len(data_val)
            val_mae /= len(data_val)
            val_r2 = np.mean(r2s)

            losses.append(
                (epoch, train_mse, train_mae, val_mse, val_mae)
            )

            if val_mae < best_mae:
                best_mae = val_mae
                best_state = model.state_dict()

            if epoch % hyperparams["print_interval"] == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train MSE {train_mse:.4f} | "
                    f"Train MAE {train_mae:.4f} | "
                    f"Val MSE {val_mse:.4f} | "
                    f"Val MAE {val_mae:.4f} | "
                    f"R² {val_r2:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest model restored (Val MAE = {best_mae:.4f})")

    print(f"--- Finished ST Training for '{name_prefix}' ---")
    return model, losses