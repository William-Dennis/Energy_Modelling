"""Challenge strategy submissions.

All ``BacktestStrategy`` subclasses placed in this package are
automatically discovered by the dashboard's Challenge tab.
"""

from strategies.always_long import AlwaysLongStrategy
from strategies.always_short import AlwaysShortStrategy
from strategies.bayesian_ridge import BayesianRidgeStrategy
from strategies.boosted_spread_ml import BoostedSpreadMLStrategy
from strategies.carbon_trend import CarbonTrendStrategy
from strategies.commodity_cost import CommodityCostStrategy
from strategies.composite_signal import CompositeSignalStrategy
from strategies.consensus_signal import ConsensusSignalStrategy
from strategies.cross_border_spread import CrossBorderSpreadStrategy
from strategies.day_of_week import DayOfWeekStrategy
from strategies.de_fr_spread import DEFRSpreadStrategy
from strategies.de_nl_spread import DENLSpreadStrategy
from strategies.decision_tree_direction import DecisionTreeStrategy
from strategies.diversity_ensemble import DiversityEnsembleStrategy
from strategies.dow_composite import DowCompositeStrategy
from strategies.elastic_net import ElasticNetStrategy
from strategies.fossil_dispatch import FossilDispatchStrategy
from strategies.fr_flow_signal import FRFlowSignalStrategy
from strategies.fuel_index_trend import FuelIndexTrendStrategy
from strategies.gas_carbon_joint_trend import GasCarbonJointTrendStrategy
from strategies.gas_trend import GasTrendStrategy
from strategies.gbm_net_demand import GBMNetDemandStrategy
from strategies.gradient_boosting_direction import GradientBoostingStrategy
from strategies.knn_direction import KNNDirectionStrategy
from strategies.lag2_reversion import Lag2ReversionStrategy
from strategies.lasso_calendar_augmented import LassoCalendarAugmentedStrategy
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.lasso_top_features import LassoTopFeaturesStrategy
from strategies.load_forecast import LoadForecastStrategy
from strategies.load_surprise import LoadSurpriseStrategy
from strategies.logistic_direction import LogisticDirectionStrategy
from strategies.majority_vote_ml import MajorityVoteMLStrategy
from strategies.majority_vote_rule import MajorityVoteRuleBasedStrategy
from strategies.mean_forecast_regression import MeanForecastRegressionStrategy
from strategies.median_forecast_ensemble import MedianForecastEnsembleStrategy
from strategies.monday_effect import MondayEffectStrategy
from strategies.month_seasonal import MonthSeasonalStrategy
from strategies.multi_spread import MultiSpreadStrategy
from strategies.net_demand import NetDemandStrategy
from strategies.net_demand_momentum import NetDemandMomentumStrategy
from strategies.neural_net import NeuralNetStrategy
from strategies.nl_flow_signal import NLFlowSignalStrategy
from strategies.nuclear_availability import NuclearAvailabilityStrategy
from strategies.pls_regression import PLSRegressionStrategy
from strategies.price_min_reversion import PriceMinReversionStrategy
from strategies.price_zscore_reversion import PriceZScoreReversionStrategy
from strategies.quarter_seasonal import QuarterSeasonalStrategy
from strategies.random_forest_direction import RandomForestStrategy
from strategies.regime_conditional_ensemble import RegimeConditionalEnsembleStrategy
from strategies.renewable_regime import RenewableRegimeStrategy
from strategies.renewables_penetration import RenewablesPenetrationStrategy
from strategies.renewables_surplus import RenewablesSurplusStrategy
from strategies.ridge_net_demand import RidgeNetDemandStrategy
from strategies.ridge_regression import RidgeRegressionStrategy
from strategies.solar_forecast import SolarForecastStrategy
from strategies.stacked_ridge_meta import StackedRidgeMetaStrategy
from strategies.svm_direction import SVMDirectionStrategy
from strategies.temperature_extreme import TemperatureExtremeStrategy
from strategies.top_k_ensemble import TopKEnsembleStrategy
from strategies.volatility_regime import VolatilityRegimeStrategy
from strategies.volatility_regime_ml import VolatilityRegimeMLStrategy
from strategies.weekday_weekend_ensemble import WeekdayWeekendEnsembleStrategy
from strategies.weekly_cycle import WeeklyCycleStrategy
from strategies.weighted_vote_mixed import WeightedVoteMixedStrategy
from strategies.wind_forecast import WindForecastStrategy
from strategies.wind_forecast_error import WindForecastErrorStrategy
from strategies.zscore_momentum import ZScoreMomentumStrategy

__all__ = [
    "AlwaysLongStrategy",
    "BoostedSpreadMLStrategy",
    "ConsensusSignalStrategy",
    "DiversityEnsembleStrategy",
    "MajorityVoteMLStrategy",
    "MajorityVoteRuleBasedStrategy",
    "MeanForecastRegressionStrategy",
    "MedianForecastEnsembleStrategy",
    "RegimeConditionalEnsembleStrategy",
    "StackedRidgeMetaStrategy",
    "TopKEnsembleStrategy",
    "WeekdayWeekendEnsembleStrategy",
    "WeightedVoteMixedStrategy",
    "AlwaysShortStrategy",
    "BayesianRidgeStrategy",
    "CarbonTrendStrategy",
    "CommodityCostStrategy",
    "CompositeSignalStrategy",
    "CrossBorderSpreadStrategy",
    "DayOfWeekStrategy",
    "DEFRSpreadStrategy",
    "DENLSpreadStrategy",
    "DecisionTreeStrategy",
    "DowCompositeStrategy",
    "ElasticNetStrategy",
    "FossilDispatchStrategy",
    "FRFlowSignalStrategy",
    "FuelIndexTrendStrategy",
    "GasCarbonJointTrendStrategy",
    "GasTrendStrategy",
    "GBMNetDemandStrategy",
    "GradientBoostingStrategy",
    "KNNDirectionStrategy",
    "Lag2ReversionStrategy",
    "LassoCalendarAugmentedStrategy",
    "LassoRegressionStrategy",
    "LassoTopFeaturesStrategy",
    "LoadForecastStrategy",
    "LoadSurpriseStrategy",
    "LogisticDirectionStrategy",
    "MondayEffectStrategy",
    "MonthSeasonalStrategy",
    "MultiSpreadStrategy",
    "NetDemandMomentumStrategy",
    "NetDemandStrategy",
    "NeuralNetStrategy",
    "NLFlowSignalStrategy",
    "NuclearAvailabilityStrategy",
    "PLSRegressionStrategy",
    "PriceMinReversionStrategy",
    "PriceZScoreReversionStrategy",
    "QuarterSeasonalStrategy",
    "RandomForestStrategy",
    "RenewableRegimeStrategy",
    "RenewablesPenetrationStrategy",
    "RenewablesSurplusStrategy",
    "RidgeNetDemandStrategy",
    "RidgeRegressionStrategy",
    "SolarForecastStrategy",
    "SVMDirectionStrategy",
    "TemperatureExtremeStrategy",
    "VolatilityRegimeMLStrategy",
    "VolatilityRegimeStrategy",
    "WeeklyCycleStrategy",
    "WindForecastStrategy",
    "WindForecastErrorStrategy",
    "ZScoreMomentumStrategy",
]
