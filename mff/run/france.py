import os
import pickle

import numpy as np
import pandas as pd
from string import ascii_uppercase, ascii_lowercase
from sktime.datasets import load_macroeconomic

from mff import MFF
from mff import MFF_mixed_freqency

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def france(): 
    df_true = pd.read_excel(r'mff\data\france\final_data.xlsx', index_col=0)

    # data transformation 
    df_true['NGDP_growth'] = df_true['NGDP'].pct_change().dropna() * 100
    df_true['NGDP_R_growth'] = df_true['NGDP_R'].pct_change().dropna() * 100
    for col in df_true.columns:
        if col not in ['NGDP', 'NGDP_R', 'NGDP_growth', 'NGDP_R_growth']:
            df_true[col] = (df_true[col] - df_true[col].shift(1)) / df_true['NGDP'].shift(1)
    df_true.drop(columns=['NGDP', 'NGDP_R'], inplace=True)
    df_true = df_true.iloc[1:]
    
    # input dataframe
    df = df_true.copy()
    fh = 7                          # forecast horizon
    ui = [i for i in range(0,15)]   # unknown variables index 
    df.iloc[-fh:,ui] = np.nan

    print("\nInput DataFrame Head (df):\n", df.head())  
    print("\nInput DataFrame Tail (showing missing values):\n", df.tail(fh+1))

    # constraints 
    equality_constraints = ['NGDP_growth?-NTDD?-NFB?-NSDGDP?','NC?-NCG?-NCP?', 
                            'NX?-NXG?-NXM?', 'NM?-NMG?-NMS?', 'NTDD?-NC?-NI?',
                            'NFB?-NX?+NM?', 'NI?-NFI?-NINV?', 'NGS?-NI?-bca?']
    
    # apply MFF
    m = MFF(df,equality_constraints=equality_constraints)
    df2 = m.fit()
    df0 = m.df0
    df1 = m.df1
    df1_model = m.df1_model
    shrinkage = m.shrinkage
    smoothness = m.smoothness
    W = m.W
    for ri,ci in np.argwhere(df.isna()):
        print(df1_model.index[ri],
              df1_model.columns[ci],
              df1_model.iloc[ri,ci].get_params())

    # print forecasted unknown variables
    print("\nForecasted values:\n", df2.iloc[-fh:,ui])

    # confirm constraints
    # assert(np.isclose(df2['NGDP_growth'] - df2['NTDD'] - df2['NFB'] - df2['NSDGDP'], 0).all())
    # assert(np.isclose(df2['NC'] - df2['NCG'] - df2['NCP'], 0).all())
    # assert(np.isclose(df2['NX'] - df2['NXG'] - df2['NXM'], 0).all())
    # assert(np.isclose(df2['NM'] - df2['NMG'] - df2['NMS'], 0).all())
    # assert(np.isclose(df2['NTDD'] - df2['NC'] - df2['NI'], 0).all())
    # assert(np.isclose(df2['NFB'] - df2['NX'] + df2['NM'], 0).all())
    # assert(np.isclose(df2['NI'] - df2['NFI'] - df2['NINV'], 0).all())
    # assert(np.isclose(df2['NGS'] - df2['NI'] - df2['bca'], 0).all())

    return {
        'df_true': df_true,
        'df': df,
        'df2': df2,
        'df0': df0,
        'df1': df1,
        'df1_model': df1_model,
        'shrinkage': shrinkage,
        'smoothness': smoothness,
        'W': W,
        'fh': fh,
        'ui': ui
    }


def plot_forecast_results(data):
    df_true = data['df_true']
    df1 = data['df1'] 
    df2 = data['df2']
    fh = data['fh']
    ui = data['ui']
    var_names = [df_true.columns[i] for i in ui]
    rmse_df1 = {df_true.columns[i]: np.sqrt(np.mean((df_true.iloc[-fh:, i] - df1.iloc[-fh:, i])**2)) for i in ui}
    rmse_df2 = {df_true.columns[i]: np.sqrt(np.mean((df_true.iloc[-fh:, i] - df2.iloc[-fh:, i])**2)) for i in ui}
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(17, 13))
    axes = axes.flatten()

    for idx, i in enumerate(ui):
        ax = axes[idx]
        ax.plot(df_true.index[-fh:], df_true.iloc[-fh:, i], label='Actual', marker='o')
        ax.plot(df_true.index[-fh:], df1.iloc[-fh:, i], label='First Stage', marker='^')
        ax.plot(df_true.index[-fh:], df2.iloc[-fh:, i], label='Predicted', marker='x')
        ax.set_title(f"{df_true.columns[i]} (RMSE: {rmse_df2[df_true.columns[i]]:.2f})", fontsize=9)
        ax.set_ylabel(" Contributions relative to GDP")
        ax.legend()
        ax.grid(True)

    ax_bar = axes[-1]
    x = np.arange(len(var_names))
    width = 0.35
    ax_bar.bar(x - width/2, [rmse_df1[col] for col in var_names], width, label='First Stage')
    ax_bar.bar(x + width/2, [rmse_df2[col] for col in var_names], width, label='Predicted')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(var_names, rotation=45, ha='right')
    ax_bar.set_title("RMSE Comparison", fontsize=9)
    ax_bar.set_ylabel("RMSE")
    ax_bar.legend(); ax_bar.grid(True)

    plt.tight_layout()
    plt.show()

def plot_rmse(data, window=5):
    df_true = data['df_true']
    df1 = data['df1']  
    df2 = data['df2'] 
    ui = data['ui']   
    mask = df_true.index >= 2016
    time_index = df_true.index[mask]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(17, 13))
    axes = axes.flatten()

    for j, i in enumerate(ui):
        col = df_true.columns[i]
        err1 = (df_true.loc[mask, col] - df1.loc[mask, col])**2
        err2 = (df_true.loc[mask, col] - df2.loc[mask, col])**2
        rmse1 = np.sqrt(err1.rolling(window=window, min_periods=1).mean())
        rmse2 = np.sqrt(err2.rolling(window=window, min_periods=1).mean())

        ax = axes[j]
        ax.plot(time_index, rmse1, label='RMSE First Stage', marker='o')
        ax.plot(time_index, rmse2, label='RMSE Predicted', marker='x')
        ax.set_title(f"Rolling RMSE: {col}")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots.
    for k in range(j + 1, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  

    forecast_file = 'forecast_data_france.pkl'
    if os.path.exists(forecast_file):
        with open(forecast_file, 'rb') as f:
            data = pickle.load(f)
        print("Loaded forecast data from file.")
    else:
        data = france()
        with open(forecast_file, 'wb') as f:
            pickle.dump(data, f)
        print("Forecast data computed and saved.")
    
    plot_forecast_results(data)
    plot_rmse(data)