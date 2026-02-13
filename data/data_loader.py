import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from config import DOWNLOAD_PREFIX, PAST_WINDOW, HORIZON, FIRST_YEAR, LAST_YEAR, TRAIN_END, VAL_END

def build_global_iso_mapping():
    edges = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/X_EDGE_1996.csv')
    iso_codes = sorted(set(edges['i']).union(set(edges['j'])))
    return {code: i for i, code in enumerate(iso_codes)}

def create_data(year, iso_code_to_id):
    edges = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/X_EDGE_{year}.csv')
    edges = edges.groupby(['i', 'j']).agg(
        {f'f{i}': 'sum' for i in range(10)}
    ).reset_index()

    edges['i_id'] = edges['i'].map(iso_code_to_id)
    edges['j_id'] = edges['j'].map(iso_code_to_id)
    edges = edges.dropna(subset=['i_id', 'j_id'])

    edge_index = torch.tensor(
        edges[['i_id', 'j_id']].values,
        dtype=torch.long
    ).t()

    EDGE_FEATURES = [f'f{i}' for i in range(10)]
    edge_attr = torch.tensor(
        edges[EDGE_FEATURES].values,
        dtype=torch.float32
    )
    edge_attr = (edge_attr - edge_attr.mean(0)) / edge_attr.std(0)
    edge_attr = torch.nan_to_num(edge_attr)

    N = len(iso_code_to_id)
    FEATURES = ['pop', 'cpi', 'emp', 'lagged_gdp']
    F = len(FEATURES)
    x_full = np.zeros((N, F), dtype=np.float32)

    x_df = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/X_NODE_{year}.csv')

    gdp_lag = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/Y_{year-1}.csv')
    gdp_lag.rename(columns={f'{year}': 'lagged_gdp'}, inplace=True)

    x_df = x_df.merge(
        gdp_lag[['iso_code', 'lagged_gdp']],
        on='iso_code', how='left'
    )
    x_df['lagged_gdp'] = np.log(x_df['lagged_gdp'].clip(1e-8))
    x_df.fillna(0, inplace=True)

    for _, row in x_df.iterrows():
        iso = row['iso_code']
        if iso in iso_code_to_id:
            idx = iso_code_to_id[iso]
            x_full[idx] = row[FEATURES].values

    x = torch.tensor(x_full, dtype=torch.float32)
    x = (x - x.mean(0)) / x.std(0)
    x = torch.nan_to_num(x)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

def load_multi_horizon_y(base_year, iso_code_to_id):
    N = len(iso_code_to_id)
    ys = []

    for h in range(1, HORIZON + 1):
        y_df = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/Y_{base_year + h - 1}.csv')
        y_full = np.zeros(N, dtype=np.float32)

        for _, row in y_df.iterrows():
            iso = row['iso_code']
            if iso in iso_code_to_id:
                idx = iso_code_to_id[iso]
                val = row.get(str(base_year + h), np.nan)
                if pd.notna(val) and val > 0:
                    y_full[idx] = np.log(val)

        ys.append(torch.tensor(y_full).unsqueeze(1))

    return torch.cat(ys, dim=1)

def create_st_sample(start_year, iso_code_to_id):
    graphs = [
        create_data(y, iso_code_to_id)
        for y in range(start_year, start_year + PAST_WINDOW)
    ]
    y = load_multi_horizon_y(
        start_year + PAST_WINDOW - 1,
        iso_code_to_id
    )
    return graphs, y

def get_st_datasets():
    iso_code_to_id = build_global_iso_mapping()
    train, val, test = [], [], []
    max_start = LAST_YEAR - PAST_WINDOW - HORIZON + 1

    for year in range(FIRST_YEAR, max_start + 1):
        sample = create_st_sample(year, iso_code_to_id)
        if year + PAST_WINDOW - 1 <= TRAIN_END:
            train.append(sample)
        elif year + PAST_WINDOW - 1 <= VAL_END:
            val.append(sample)
        else:
            test.append(sample)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

def unpack_graphs(graphs):
    xs = torch.stack([g.x for g in graphs], dim=0)
    edge_indices = [g.edge_index for g in graphs]
    edge_attrs = [g.edge_attr for g in graphs]
    return xs, edge_indices, edge_attrs