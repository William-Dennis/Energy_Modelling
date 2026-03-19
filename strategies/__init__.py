"""Example challenge submissions and baselines."""

from strategies.always_short import AlwaysShortStrategy
from strategies.de_france_spread_strategy import DEFranceSpreadStrategy
from strategies.gas_trend_strategy import GasTrendStrategy
from strategies.load_forecast_median_strategy import LoadForecastMedianStrategy
from strategies.price_level_mean_reversion_strategy import PriceLevelMeanReversionStrategy
from strategies.skip_all_strategy import SkipAllStrategy
from strategies.solar_forecast_contrarian_strategy import SolarForecastContrarianStrategy
from strategies.student_strategy import StudentStrategy
from strategies.tiny_ml_strategy import TinyMLStrategy
from strategies.wind_forecast_contrarian_strategy import WindForecastContrarianStrategy
from strategies.yesterday_mean_reversion_strategy import YesterdayMeanReversionStrategy
from strategies.yesterday_momentum_strategy import YesterdayMomentumStrategy

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
