from models.arima_model import train_arima as train_arima_core

def train_arima(
    name_prefix,
    hyperparams,
    data_train,
    data_val
):
    return train_arima_core(name_prefix, hyperparams, data_train, data_val)