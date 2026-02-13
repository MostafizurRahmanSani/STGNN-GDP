import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class GRUModel(nn.Module):
    def __init__(self, hidden_dim=32, horizon=3, input_dim=4, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, horizon)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

def build_gru_dataset_from_stgnn(data, past_window, horizon, feature_scaler=None, target_scaler=None, fit_scaler=False):
    Xs, Ys = [], []
    
    for graphs, y in data:
        xs, _, _ = unpack_graphs(graphs)  
        y_target = y  
        
        xs = xs.permute(1, 0, 2)  
        
        Xs.append(xs.cpu().numpy())
        Ys.append(y_target.cpu().numpy())
    
 
    X = np.concatenate(Xs, axis=0)  
    Y = np.concatenate(Ys, axis=0) 
    
    if fit_scaler:
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        X_shape = X.shape
        Y_shape = Y.shape
        X = feature_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X_shape)
        Y = target_scaler.fit_transform(Y.reshape(-1, 1)).reshape(Y_shape)
        return X, Y, feature_scaler, target_scaler
    else:
        X_shape = X.shape
        Y_shape = Y.shape
        X = feature_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X_shape)
        Y = target_scaler.transform(Y.reshape(-1, 1)).reshape(Y_shape)
        return X, Y
    
def train_gru(model, hyperparams, X_train, Y_train, X_val, Y_val, feature_scaler, target_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.get("batch_size", 32), shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0
    max_patience = hyperparams.get("patience", 50)
    losses = []
    
    for epoch in range(hyperparams["n_epochs"]):
        model.train()
        train_losses = []
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = F.mse_loss(pred, batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_mse = F.mse_loss(val_pred, Y_val_t.to(device)).item()
            val_mae = F.l1_loss(val_pred, Y_val_t.to(device)).item()
            val_r2 = r2_score(Y_val_t.cpu().numpy().reshape(-1), val_pred.cpu().numpy().reshape(-1))
        
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % hyperparams.get("save_loss_interval", 10) == 0:
            train_mse = np.mean(train_losses)
            train_mae = F.l1_loss(model(X_train_t.to(device)), Y_train_t.to(device)).item()
            losses.append((epoch, train_mse, train_mae, val_mse, val_mae))
                    
        if epoch % hyperparams.get("print_interval", 50) == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {np.mean(train_losses):.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | Val R²: {val_r2:.4f}")
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    print(f"\n Best GRU restored (Val MAE = {best_val_mae:.4f})")
    
 
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'hidden_dim': model.gru.hidden_size,
            'horizon': model.fc.out_features,
            'input_dim': model.gru.input_size,
            'num_layers': model.gru.num_layers
        },
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'best_val_mae': best_val_mae
    }, "gru_model.pt")
    print("✓ GRU model saved to gru_model.pt")
    
    return model, losses, feature_scaler, target_scaler

def evaluate_gru_test(model, X_test, Y_test, target_scaler, horizon=3):
    model.eval()
    device = next(model.parameters()).device
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        pred_scaled = model(X_test_t).cpu().numpy()
        true_scaled = Y_test
    
    pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    true_original = target_scaler.inverse_transform(true_scaled.reshape(-1, 1)).reshape(true_scaled.shape)
    
    results = {}
    for h in range(horizon):
        y_true = true_original[:, h]
        y_pred = pred_original[:, h]
        results[f"t+{h+1}"] = {
            "MSE": mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }
    
    results["aggregate"] = {
        "MSE": mean_squared_error(true_original.reshape(-1), pred_original.reshape(-1)),
        "MAE": mean_absolute_error(true_original.reshape(-1), pred_original.reshape(-1)),
        "RMSE": np.sqrt(mean_squared_error(true_original.reshape(-1), pred_original.reshape(-1))),
        "R2": r2_score(true_original.reshape(-1), pred_original.reshape(-1))
    }
    
    return results, pred_original, true_original

def unpack_graphs(graphs):
    xs = torch.stack([g.x for g in graphs], dim=0)
    edge_indices = [g.edge_index for g in graphs]
    edge_attrs = [g.edge_attr for g in graphs]
    return xs, edge_indices, edge_attrs


def build_gru_dataset(arima_data, past_window, horizon):
    Xs, Ys = [], []
    for iso, x_train, y in arima_data:
        if len(x_train) < past_window or len(y) < horizon:
            continue
        x_seq = x_train[-past_window:]
        x_seq = torch.tensor(x_seq).float().unsqueeze(-1)
        y_seq = torch.tensor(y[:horizon]).float()
        Xs.append(x_seq)
        Ys.append(y_seq)
    X = torch.stack(Xs)
    Y = torch.stack(Ys)
    return X, Y