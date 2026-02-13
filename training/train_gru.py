from models.gru_model import train_gru as train_gru_core

def train_gru(
    model,
    hyperparams,
    X_train, Y_train,
    X_val, Y_val
):
    return train_gru_core(model, hyperparams, X_train, Y_train, X_val, Y_val)