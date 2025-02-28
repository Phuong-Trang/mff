import os
import pickle
import argparse

import numpy as np
import pandas as pd
from sktime.datasets import load_macroeconomic

from mff import MFF
from mff import MFF_mixed_freqency

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def forecast_country(country, exp_number, path):
    df_true = pd.read_excel(rf'{path}', index_col=0)
    weo_forecast = df_true.pop(df_true.columns[0])
    df = df_true.copy()

    # Set forecast horizon (fh) and unknown variable indices (ui) based on country and experiment
    if country.lower() == 'singapore':
        fh = 7
        ui = list(range(1)) if exp_number == 1 else list(range(16))
    elif country.lower() == 'usa':
        # fh = 8 if exp_number == 2 else 7
        fh = 3
        ui = list(range(1)) if exp_number == 1 else list(range(16))
    elif country.lower() == 'china':
        fh = 8
        ui_options = [list(range(1)), list(range(16)), list(range(15))]
        ui = ui_options[exp_number - 1]

    df.iloc[-fh:, ui] = np.nan
    print(f"\n[{country.capitalize()}] Input DataFrame Head (df):\n", df.head())
    print(f"\n[{country.capitalize()}] Input DataFrame Tail (showing missing values):\n", df.tail(fh+1))

    # Equality constraints 
    equality_constraints = None
    if exp_number == 1:
        pass
    elif country.lower() == 'china' and exp_number == 2:
        equality_constraints = ['NGDP?-NC?-NI?-NFB?-NSDGDP?', 'NFB?-NX?+NM?',
                                'NX?-NXG?-NXM?', 'NM?-NMG?-NMS?', 'NI?-NFI?-NINV?']
    else:
        equality_constraints = ['NGDP?-NC?-NI?-NFB?-NSDGDP?', 'NFB?-NX?+NM?',
                                'NX?-NXG?-NXM?', 'NM?-NMG?-NMS?',
                                'NI?-NFI?-NINV?', 'NGS?-NI?-bca?']

    # Apply MFF 
    if equality_constraints:
        m = MFF(df, equality_constraints=equality_constraints)
    else:
        m = MFF(df)
    df2 = m.fit()
    df0 = m.df0
    df1 = m.df1
    df1_model = m.df1_model
    shrinkage = m.shrinkage
    smoothness = m.smoothness
    W = m.W

    print(f"\n[{country.capitalize()}] 1st stage forecasted values:\n", df1.iloc[-fh:, ui])
    print(f"\n[{country.capitalize()}] 1st+2nd stage forecasted values:\n", df2.iloc[-fh:, ui])

    return {
        'df_true': df_true,
        'weo_forecast': weo_forecast,
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


def plot_forecast_results(data, country, exp_number):
    df_true = data['df_true']
    df1 = data['df1']
    df2 = data['df2']
    fh = data['fh']
    ui = data['ui']
    var_names = [df_true.columns[i] for i in ui]
    rmse_df2 = {df_true.columns[i]: np.sqrt(np.mean((df_true.iloc[-fh:, i] - df2.iloc[-fh:, i])**2)) for i in ui}

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(17, 13))
    fig.canvas.manager.set_window_title(f"{country.capitalize()} Forecast Results {exp_number}")
    axes = axes.flatten()

    for j, i in enumerate(ui):
        ax = axes[j]
        ax.plot(df_true.index[-fh:], df_true.iloc[-fh:, i], label='Actual', marker='o')
        ax.plot(df_true.index[-fh:], data['df1'].iloc[-fh:, i], label='First Stage', marker='^')
        ax.plot(df_true.index[-fh:], df2.iloc[-fh:, i], label='Predicted', marker='x')
        ax.set_title(f"{df_true.columns[i]} (RMSE: {rmse_df2[df_true.columns[i]]:.2f})", fontsize=9)
        ax.set_ylabel("Billions of euros")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots.
    for k in range(j + 1, len(axes)):
        axes[k].set_visible(False)
    plt.tight_layout()

    if not os.path.exists("figures"):
        os.makedirs("figures")
    save_path = os.path.join("figures", f"{country.capitalize()} Forecast Results {exp_number}.png")
    fig.savefig(save_path)

    plt.show()


def plot_rmse(data, exp_number, window=5):
    df_true = data['df_true']
    df1 = data['df1']
    df2 = data['df2']
    weo_forecast = data['weo_forecast']
    ui = data['ui']
    mask = df_true.index >= 2021
    time_index = df_true.index[mask]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(17, 13))
    fig.canvas.manager.set_window_title(f"{country.capitalize()} SE comparision {exp_number}")
    axes = axes.flatten()

    for j, i in enumerate(ui):
        col = df_true.columns[i]
        err1 = (df_true.loc[mask, col] - df1.loc[mask, col]) ** 2
        err2 = (df_true.loc[mask, col] - df2.loc[mask, col]) ** 2
        # rmse1 = np.sqrt(err1.rolling(window=window, min_periods=1).mean())
        # rmse2 = np.sqrt(err2.rolling(window=window, min_periods=1).mean())

        ax = axes[j]
        ax.plot(time_index, err1, label='SE First Stage', marker='o')
        ax.plot(time_index, err2, label='SE Predicted', marker='x')
        print(f'SE First Stage: {err1} \n SE Predicted: {err2}')

        # For the first subplot, also compare with the original WEO forecast.
        if j == 0:
            err3 = (df_true.loc[mask, col] - weo_forecast.loc[mask]) ** 2
            # rmse3 = np.sqrt(err3.rolling(window=window, min_periods=1).mean())
            ax.plot(time_index, err3, label='SE WEO', marker='s')
            print(f'SE WEO: {err3}')

        ax.set_title(f"Rolling SE: {col}")
        ax.set_ylabel("Billions of euros")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots.
    for k in range(j + 1, len(axes)):
        axes[k].set_visible(False)
    plt.tight_layout()

    if not os.path.exists("figures"):
        os.makedirs("figures")
    save_path = os.path.join("figures", f"{country.capitalize()} SE comparision {exp_number}.png")
    fig.savefig(save_path)

    plt.show()


def call_forecast(country):
    # Define file names and paths for each country.
    forecast_files = [f'forecast_data_{country}{i}.pkl' for i in range(1, 4)]
    data_paths = [
            rf'E:\macroframe-forecast\mff\data\{country}\exp1_onlyWEO_data.xlsx',
            rf'E:\macroframe-forecast\mff\data\{country}\exp2_addUnknownVars_moreYearObservations_lessKnownVars.xlsx',
            rf'E:\macroframe-forecast\mff\data\{country}\exp3_addUnknownVars_lessYearObservations_moreKnownVars.xlsx'
        ]

    # Run forecast for each experiment associated with the country.
    for idx, file in enumerate(forecast_files):
        if os.path.exists(file):
            with open(file, 'rb') as f:
                data = pickle.load(f)
            print(f"[{country.capitalize()}] Loaded forecast data from file: {file}")
        else:
            data = forecast_country(country, idx + 1, data_paths[idx])
            with open(file, 'wb') as f:
                pickle.dump(data, f)
            print(f"[{country.capitalize()}] Forecast data computed and saved to: {file}")

        plot_forecast_results(data, country, os.path.basename(data_paths[idx]))
        plot_rmse(data, os.path.basename(data_paths[idx]))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # Run forecasts for all supported countries sequentially.
    for country in ['usa']:
        print(f"\n================ Running Forecasts for {country.capitalize()} ================\n")
        call_forecast(country)

