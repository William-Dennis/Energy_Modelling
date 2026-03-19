"""Example challenge submissions and baselines."""

from submission.always_short_strategy import AlwaysShortStrategy
from submission.de_france_spread_strategy import DEFranceSpreadStrategy
from submission.gas_trend_strategy import GasTrendStrategy
from submission.load_forecast_median_strategy import LoadForecastMedianStrategy
from submission.price_level_mean_reversion_strategy import PriceLevelMeanReversionStrategy
from submission.skip_all_strategy import SkipAllStrategy
from submission.solar_forecast_contrarian_strategy import SolarForecastContrarianStrategy
from submission.student_strategy import StudentStrategy
from submission.tiny_ml_strategy import TinyMLStrategy
from submission.wind_forecast_contrarian_strategy import WindForecastContrarianStrategy
from submission.yesterday_mean_reversion_strategy import YesterdayMeanReversionStrategy
from submission.yesterday_momentum_strategy import YesterdayMomentumStrategy

__all__ = [
    "AlwaysShortStrategy",
    "DEFranceSpreadStrategy",
    "GasTrendStrategy",
    "LoadForecastMedianStrategy",
    "PriceLevelMeanReversionStrategy",
    "SkipAllStrategy",
    "SolarForecastContrarianStrategy",
    "StudentStrategy",
    "TinyMLStrategy",
    "WindForecastContrarianStrategy",
    "YesterdayMeanReversionStrategy",
    "YesterdayMomentumStrategy",
]
