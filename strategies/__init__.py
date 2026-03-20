"""Challenge strategy submissions.

All ``BacktestStrategy`` subclasses placed in this package are
automatically discovered by the dashboard's Challenge tab.
"""

from strategies.always_long import AlwaysLongStrategy
from strategies.always_short import AlwaysShortStrategy
from strategies.composite_signal import CompositeSignalStrategy
from strategies.day_of_week import DayOfWeekStrategy
from strategies.dow_composite import DowCompositeStrategy
from strategies.fossil_dispatch import FossilDispatchStrategy
from strategies.lag2_reversion import Lag2ReversionStrategy
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.load_forecast import LoadForecastStrategy
from strategies.weekly_cycle import WeeklyCycleStrategy
from strategies.wind_forecast import WindForecastStrategy

__all__ = [
    "AlwaysLongStrategy",
    "AlwaysShortStrategy",
    "CompositeSignalStrategy",
    "DayOfWeekStrategy",
    "DowCompositeStrategy",
    "FossilDispatchStrategy",
    "Lag2ReversionStrategy",
    "LassoRegressionStrategy",
    "LoadForecastStrategy",
    "WeeklyCycleStrategy",
    "WindForecastStrategy",
]
