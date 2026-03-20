"""Challenge strategy submissions.

All ``BacktestStrategy`` subclasses placed in this package are
automatically discovered by the dashboard's Challenge tab.
"""

from strategies.always_long import AlwaysLongStrategy
from strategies.always_short import AlwaysShortStrategy
from strategies.carbon_trend import CarbonTrendStrategy
from strategies.commodity_cost import CommodityCostStrategy
from strategies.composite_signal import CompositeSignalStrategy
from strategies.cross_border_spread import CrossBorderSpreadStrategy
from strategies.day_of_week import DayOfWeekStrategy
from strategies.de_fr_spread import DEFRSpreadStrategy
from strategies.de_nl_spread import DENLSpreadStrategy
from strategies.dow_composite import DowCompositeStrategy
from strategies.fossil_dispatch import FossilDispatchStrategy
from strategies.fr_flow_signal import FRFlowSignalStrategy
from strategies.fuel_index_trend import FuelIndexTrendStrategy
from strategies.gas_trend import GasTrendStrategy
from strategies.lag2_reversion import Lag2ReversionStrategy
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.load_forecast import LoadForecastStrategy
from strategies.load_surprise import LoadSurpriseStrategy
from strategies.multi_spread import MultiSpreadStrategy
from strategies.net_demand import NetDemandStrategy
from strategies.nl_flow_signal import NLFlowSignalStrategy
from strategies.nuclear_availability import NuclearAvailabilityStrategy
from strategies.price_min_reversion import PriceMinReversionStrategy
from strategies.price_zscore_reversion import PriceZScoreReversionStrategy
from strategies.renewables_penetration import RenewablesPenetrationStrategy
from strategies.renewables_surplus import RenewablesSurplusStrategy
from strategies.solar_forecast import SolarForecastStrategy
from strategies.temperature_extreme import TemperatureExtremeStrategy
from strategies.volatility_regime import VolatilityRegimeStrategy
from strategies.weekly_cycle import WeeklyCycleStrategy
from strategies.wind_forecast import WindForecastStrategy
from strategies.wind_forecast_error import WindForecastErrorStrategy

__all__ = [
    "AlwaysLongStrategy",
    "AlwaysShortStrategy",
    "CarbonTrendStrategy",
    "CommodityCostStrategy",
    "CompositeSignalStrategy",
    "CrossBorderSpreadStrategy",
    "DayOfWeekStrategy",
    "DEFRSpreadStrategy",
    "DENLSpreadStrategy",
    "DowCompositeStrategy",
    "FossilDispatchStrategy",
    "FRFlowSignalStrategy",
    "FuelIndexTrendStrategy",
    "GasTrendStrategy",
    "Lag2ReversionStrategy",
    "LassoRegressionStrategy",
    "LoadForecastStrategy",
    "LoadSurpriseStrategy",
    "MultiSpreadStrategy",
    "NetDemandStrategy",
    "NLFlowSignalStrategy",
    "NuclearAvailabilityStrategy",
    "PriceMinReversionStrategy",
    "PriceZScoreReversionStrategy",
    "RenewablesPenetrationStrategy",
    "RenewablesSurplusStrategy",
    "SolarForecastStrategy",
    "TemperatureExtremeStrategy",
    "VolatilityRegimeStrategy",
    "WeeklyCycleStrategy",
    "WindForecastStrategy",
    "WindForecastErrorStrategy",
]
