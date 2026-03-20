"""Challenge strategy submissions.

All ``BacktestStrategy`` subclasses placed in this package are
automatically discovered by the dashboard's Challenge tab.
"""

from strategies.always_long import AlwaysLongStrategy
from strategies.always_short import AlwaysShortStrategy
from strategies.commodity_cost import CommodityCostStrategy
from strategies.composite_signal import CompositeSignalStrategy
from strategies.cross_border_spread import CrossBorderSpreadStrategy
from strategies.day_of_week import DayOfWeekStrategy
from strategies.dow_composite import DowCompositeStrategy
from strategies.fossil_dispatch import FossilDispatchStrategy
from strategies.lag2_reversion import Lag2ReversionStrategy
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.load_forecast import LoadForecastStrategy
from strategies.nuclear_availability import NuclearAvailabilityStrategy
from strategies.renewables_surplus import RenewablesSurplusStrategy
from strategies.solar_forecast import SolarForecastStrategy
from strategies.temperature_extreme import TemperatureExtremeStrategy
from strategies.volatility_regime import VolatilityRegimeStrategy
from strategies.weekly_cycle import WeeklyCycleStrategy
from strategies.wind_forecast import WindForecastStrategy

__all__ = [
    "AlwaysLongStrategy",
    "AlwaysShortStrategy",
    "CommodityCostStrategy",
    "CompositeSignalStrategy",
    "CrossBorderSpreadStrategy",
    "DayOfWeekStrategy",
    "DowCompositeStrategy",
    "FossilDispatchStrategy",
    "Lag2ReversionStrategy",
    "LassoRegressionStrategy",
    "LoadForecastStrategy",
    "NuclearAvailabilityStrategy",
    "RenewablesSurplusStrategy",
    "SolarForecastStrategy",
    "TemperatureExtremeStrategy",
    "VolatilityRegimeStrategy",
    "WeeklyCycleStrategy",
    "WindForecastStrategy",
]
