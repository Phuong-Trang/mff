import os
import pickle
import argparse

import numpy as np
import pandas as pd
from sktime.datasets import load_macroeconomic

from mff import MFF
from mff import MFF_mixed_freqency

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error


def forecast_country(country, path):
    df_true = pd.read_excel(rf'{path}', index_col=0)
    weo_forecast = df_true.pop(df_true.columns[0])
    df = df_true.copy()
    if '16' in country:
        fh = 8
    else:
        fh = 3
    ui = list(range(16))
    df.iloc[-fh:, ui] = np.nan
    print(f"\n[{country.capitalize()}] Input DataFrame Head (df):\n", df.head())
    print(f"\n[{country.capitalize()}] Input DataFrame Tail (showing missing values):\n", df.tail(fh+1))

    # Equality constraints 
    equality_constraints = ['NGDP_R?-NC?-NI?-NFB?-NSDGDP?', 'NFB?-NX?+NM?',
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


def plot_forecast_results(data, country_code):
    df_true = data['df_true']
    df1 = data['df1']
    df2 = data['df2']
    weo = data['weo_forecast']
    fh = data['fh']
    ui = data['ui']
    rmse_df1 = {df_true.columns[i]: np.sqrt(np.mean((df_true.iloc[-fh:, i] - df1.iloc[-fh:, i])**2)) for i in ui}
    rmse_df2 = {df_true.columns[i]: np.sqrt(np.mean((df_true.iloc[-fh:, i] - df2.iloc[-fh:, i])**2)) for i in ui}
    rmse_weo = {df_true.columns[i]: np.sqrt(np.mean((df_true.iloc[-fh:, i] - weo.iloc[-fh:])**2)) for i in ui}

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(17, 13))
    fig.canvas.manager.set_window_title(f"{country_code.capitalize()} Forecast Results")
    axes = axes.flatten()

    for j, i in enumerate(ui):
        ax = axes[j]
        ax.plot(df_true.index[-fh:], df_true.iloc[-fh:, i], label='Actual', marker='o')
        ax.plot(df_true.index[-fh:], df1.iloc[-fh:, i], label='First Stage', marker='^')
        ax.plot(df_true.index[-fh:], df2.iloc[-fh:, i], label='Predicted', marker='x')
        ax.set_title(f"{df_true.columns[i]} (RMSE: {rmse_df2[df_true.columns[i]]:.2f})", fontsize=9)
        ax.set_ylabel("Billions of euros")
        ax.legend()
        ax.grid(True)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if j == 0: 
            print(f"\n1st stage forecasted values:\n", df1.iloc[-fh:, ui])
            print(f"\n1st+2nd stage forecasted values:\n", df2.iloc[-fh:, ui])
            print(f'\nRMSE NGDP_R: \nAfter 1st stage: {rmse_df1[df_true.columns[i]]:.2f} \nAfter 2nd stage: {rmse_df2[df_true.columns[i]]:.2f} \nWEO: {rmse_weo[df_true.columns[i]]:.2f}')

    # Hide any unused subplots.
    for k in range(j + 1, len(axes)):
        axes[k].set_visible(False)
    plt.tight_layout()

    if not os.path.exists(r"E:\macroframe-forecast\NGDPR_forecast\figures"):
        os.makedirs(r"E:\macroframe-forecast\NGDPR_forecast\figures")
    save_path = os.path.join(r"E:\macroframe-forecast\NGDPR_forecast\figures", f"{country_code.capitalize()} Forecast Results.png")
    fig.savefig(save_path)

    plt.show()


def plot_se(data, country_code):
    df_true = data['df_true']
    df1 = data['df1']
    df2 = data['df2']
    weo = data['weo_forecast']
    ui = data['ui']
    if '16' in country_code:
        mask = df_true.index >= 2016
    else:
        mask = df_true.index >= 2021
    time_index = df_true.index[mask]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(17, 13))
    fig.canvas.manager.set_window_title(f"{country_code.capitalize()} SE comparision")
    axes = axes.flatten()

    for j, i in enumerate(ui):
        col = df_true.columns[i]
        err1 = (df_true.loc[mask, col] - df1.loc[mask, col]) ** 2
        err2 = (df_true.loc[mask, col] - df2.loc[mask, col]) ** 2

        ax = axes[j]
        ax.plot(time_index, err1, label='After 1st stage', marker='o')
        ax.plot(time_index, err2, label='After 2nd stage', marker='x')

        # For the first subplot, also compare with the original WEO forecast.
        if j == 0:
            err3 = (df_true.loc[mask, col] - weo.loc[mask]) ** 2
            ax.plot(time_index, err3, label='SE WEO', marker='s')
            print(f'SE NGDP_R: \nAfter 1st stage: {err1} \nAfter 2nd stage: {err2} \nWEO: {err3}')

        ax.set_title(f"Rolling SE: {col}")
        ax.set_ylabel("Billions of euros")
        ax.legend()
        ax.grid(True)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Hide any unused subplots.
    for k in range(j + 1, len(axes)):
        axes[k].set_visible(False)
    plt.tight_layout()

    if not os.path.exists(r"E:\macroframe-forecast\NGDPR_forecast\figures"):
        os.makedirs(r"E:\macroframe-forecast\NGDPR_forecast\figures")
    save_path = os.path.join(r"E:\macroframe-forecast\NGDPR_forecast\figures", f"{country_code.capitalize()} SE comparision.png")
    fig.savefig(save_path)

    plt.show()


def call_forecast(country_code):
    # Define file names and paths for each country.
    file = rf'E:\macroframe-forecast\NGDPR_forecast\forecasts\forecast_data_{country_code}.pkl'
    data_path = rf'E:\macroframe-forecast\NGDPR_forecast\data\{country_code}.xlsx'

    # Run forecast for each experiment associated with the country.
    if os.path.exists(file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        print(f"[{country_code.capitalize()}] Loaded forecast data from file: {file}")
    else:
        data = forecast_country(country_code, data_path)
        with open(file, 'wb') as f:
            pickle.dump(data, f)
        print(f"[{country_code.capitalize()}] Forecast data computed and saved to: {file}")

    plot_forecast_results(data, country_code)
    plot_se(data, country_code)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # Run forecasts for all supported countries sequentially.
    for country_code in [f"{c}{y}" for y in ['16', '21'] for c in ['sgp', 'usa', 'chn']]:
        print(f"\n================ Running Forecasts for {country_code.capitalize()} ================\n")
        call_forecast(country_code)

