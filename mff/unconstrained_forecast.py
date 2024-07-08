from typing import Optional, Tuple

from pandas import DataFrame
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

from .get_default_forecaster import get_default_forecaster



import cProfile
import pstats
import io
from functools import wraps
def profile(func):
    """A decorator that uses cProfile to profile a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        return result

    return wrapper


#@profile
def unconstrained_forecast(
    df: DataFrame, Tin: int, forecaster: Optional[BaseForecaster] = None, fh: ForecastingHorizon or int = None
) -> tuple[DataFrame, BaseForecaster, ForecastingHorizon]:
    if forecaster is None:
        forecaster = get_default_forecaster(Tin=Tin)

    if not df.isna().sum(axis=1).all():
        yf = df
        Xf = None
        Xp = None
    else:
        unknown_variables = df.columns[df.isna().sum(axis=0) > 0]
        known_variables = df.columns[df.isna().sum(axis=0) == 0]
        mask_fit = df[unknown_variables].notna().sum(axis=1) > 0
        mask_predict = df[unknown_variables].isna().sum(axis=1) > 0

        Xf = df.loc[mask_fit, known_variables]
        yf = df.loc[mask_fit, unknown_variables]

        Xp = df.loc[mask_predict, known_variables].reset_index(drop=True)

    # yp = df.loc[mask_predict, unknown_variables]
    if isinstance(fh, int):
        fh = ForecastingHorizon(values=range(1, fh + 1), is_relative=True)
    elif isinstance(fh, ForecastingHorizon):
        fh = fh
    else:
        fh = ForecastingHorizon(values=Xp.index + 1, is_relative=True)

    yp = forecaster.fit_predict(y=yf, X=Xf, fh=fh, X_pred=Xp)

    df1 = df.copy()
    df1 = df1.append(yp) #TODO: handle when some exogenous are present

    return df1, forecaster, fh
