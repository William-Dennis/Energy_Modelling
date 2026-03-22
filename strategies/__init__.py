"""Challenge strategy submissions.

All ``BacktestStrategy`` subclasses placed in this package are
automatically discovered by the dashboard's Challenge tab.
"""

from strategies.always_long import AlwaysLongStrategy
from strategies.always_short import AlwaysShortStrategy
from strategies.bayesian_ridge import BayesianRidgeStrategy
from strategies.boosted_spread_ml import BoostedSpreadMLStrategy
from strategies.carbon_gas_ratio import CarbonGasRatioStrategy
from strategies.carbon_trend import CarbonTrendStrategy
from strategies.commodity_cost import CommodityCostStrategy
from strategies.composite_signal import CompositeSignalStrategy
from strategies.consensus_signal import ConsensusSignalStrategy
from strategies.contrarian_momentum import ContrarianMomentumStrategy
from strategies.conviction_weighted import ConvictionWeightedStrategy
from strategies.cross_border_spread import CrossBorderSpreadStrategy
from strategies.czech_austrian_mean import CzechAustrianMeanStrategy
from strategies.day_of_week import DayOfWeekStrategy
from strategies.de_fr_spread import DEFRSpreadStrategy
from strategies.de_nl_spread import DENLSpreadStrategy
from strategies.decision_tree_direction import DecisionTreeStrategy
from strategies.denmark_spread import DenmarkSpreadStrategy
from strategies.diversity_ensemble import DiversityEnsembleStrategy
from strategies.dow_composite import DowCompositeStrategy
from strategies.elastic_net import ElasticNetStrategy
from strategies.flow_imbalance import FlowImbalanceStrategy
from strategies.forecast_price_error import ForecastPriceErrorStrategy
from strategies.fossil_dispatch import FossilDispatchStrategy
from strategies.fr_flow_signal import FRFlowSignalStrategy
from strategies.fuel_index_trend import FuelIndexTrendStrategy
from strategies.gas_carbon_joint_trend import GasCarbonJointTrendStrategy
from strategies.gas_trend import GasTrendStrategy
from strategies.gbm_net_demand import GBMNetDemandStrategy
from strategies.gradient_boosting_direction import GradientBoostingStrategy
from strategies.high_vol_skip import HighVolSkipStrategy
from strategies.independent_vote import IndependentVoteStrategy
from strategies.intraday_range import IntradayRangeStrategy
from strategies.knn_direction import KNNDirectionStrategy
from strategies.lag2_reversion import Lag2ReversionStrategy
from strategies.lasso_calendar_augmented import LassoCalendarAugmentedStrategy
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.lasso_top_features import LassoTopFeaturesStrategy
from strategies.load_forecast import LoadForecastStrategy
from strategies.load_generation_gap import LoadGenerationGapStrategy
from strategies.load_surprise import LoadSurpriseStrategy
from strategies.logistic_direction import LogisticDirectionStrategy
from strategies.majority_vote_ml import MajorityVoteMLStrategy
from strategies.majority_vote_rule import MajorityVoteRuleBasedStrategy
from strategies.mean_forecast_regression import MeanForecastRegressionStrategy
from strategies.median_forecast_ensemble import MedianForecastEnsembleStrategy
from strategies.median_independent import MedianIndependentStrategy
from strategies.monday_effect import MondayEffectStrategy
from strategies.month_seasonal import MonthSeasonalStrategy
from strategies.monthly_mean_reversion import MonthlyMeanReversionStrategy
from strategies.multi_spread import MultiSpreadStrategy
from strategies.net_demand import NetDemandStrategy
from strategies.net_demand_momentum import NetDemandMomentumStrategy
from strategies.neural_net import NeuralNetStrategy
from strategies.nl_flow_signal import NLFlowSignalStrategy
from strategies.nuclear_availability import NuclearAvailabilityStrategy
from strategies.nuclear_event import NuclearEventStrategy
from strategies.nuclear_gas_substitution import NuclearGasSubstitutionStrategy
from strategies.offshore_wind_anomaly import OffshoreWindAnomalyStrategy
from strategies.pls_regression import PLSRegressionStrategy
from strategies.poland_spread import PolandSpreadStrategy
from strategies.price_min_reversion import PriceMinReversionStrategy
from strategies.price_zscore_reversion import PriceZScoreReversionStrategy
from strategies.pruned_ml_ensemble import PrunedMLEnsembleStrategy
from strategies.quarter_seasonal import QuarterSeasonalStrategy
from strategies.radiation_regime import RadiationRegimeStrategy
from strategies.radiation_solar import RadiationSolarStrategy
from strategies.random_forest_direction import RandomForestStrategy
from strategies.regime_conditional_ensemble import RegimeConditionalEnsembleStrategy
from strategies.regime_ridge import RegimeRidgeStrategy
from strategies.renewable_ramp import RenewableRampStrategy
from strategies.renewable_regime import RenewableRegimeStrategy
from strategies.renewables_penetration import RenewablesPenetrationStrategy
from strategies.renewables_surplus import RenewablesSurplusStrategy
from strategies.ridge_net_demand import RidgeNetDemandStrategy
from strategies.ridge_regression import RidgeRegressionStrategy
from strategies.seasonal_regime_switch import SeasonalRegimeSwitchStrategy
from strategies.selective_high_conviction import SelectiveHighConvictionStrategy
from strategies.solar_forecast import SolarForecastStrategy
from strategies.spark_spread import SparkSpreadStrategy
from strategies.spread_consensus import SpreadConsensusStrategy
from strategies.spread_momentum import SpreadMomentumStrategy
from strategies.stacked_ridge_meta import StackedRidgeMetaStrategy
from strategies.supply_demand_balance import SupplyDemandBalanceStrategy
from strategies.svm_direction import SVMDirectionStrategy
from strategies.temperature_curve import TemperatureCurveStrategy
from strategies.temperature_extreme import TemperatureExtremeStrategy
from strategies.top_k_ensemble import TopKEnsembleStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from strategies.volatility_regime import VolatilityRegimeStrategy
from strategies.volatility_regime_ml import VolatilityRegimeMLStrategy
from strategies.weekday_weekend_ensemble import WeekdayWeekendEnsembleStrategy
from strategies.weekend_mean_reversion import WeekendMeanReversionStrategy
from strategies.weekly_autocorrelation import WeeklyAutocorrelationStrategy
from strategies.weekly_cycle import WeeklyCycleStrategy
from strategies.weighted_vote_mixed import WeightedVoteMixedStrategy
from strategies.wind_forecast import WindForecastStrategy
from strategies.wind_forecast_error import WindForecastErrorStrategy
from strategies.zscore_momentum import ZScoreMomentumStrategy

__all__ = [
    "AlwaysLongStrategy",
    "AlwaysShortStrategy",
    "BayesianRidgeStrategy",
    "BoostedSpreadMLStrategy",
    "CarbonGasRatioStrategy",
    "CarbonTrendStrategy",
    "CommodityCostStrategy",
    "CompositeSignalStrategy",
    "ConsensusSignalStrategy",
    "ContrarianMomentumStrategy",
    "ConvictionWeightedStrategy",
    "CrossBorderSpreadStrategy",
    "CzechAustrianMeanStrategy",
    "DayOfWeekStrategy",
    "DEFRSpreadStrategy",
    "DENLSpreadStrategy",
    "DecisionTreeStrategy",
    "DenmarkSpreadStrategy",
    "DiversityEnsembleStrategy",
    "DowCompositeStrategy",
    "ElasticNetStrategy",
    "FlowImbalanceStrategy",
    "ForecastPriceErrorStrategy",
    "FossilDispatchStrategy",
    "FRFlowSignalStrategy",
    "FuelIndexTrendStrategy",
    "GasCarbonJointTrendStrategy",
    "GasTrendStrategy",
    "GBMNetDemandStrategy",
    "GradientBoostingStrategy",
    "HighVolSkipStrategy",
    "IndependentVoteStrategy",
    "IntradayRangeStrategy",
    "KNNDirectionStrategy",
    "Lag2ReversionStrategy",
    "LassoCalendarAugmentedStrategy",
    "LassoRegressionStrategy",
    "LassoTopFeaturesStrategy",
    "LoadForecastStrategy",
    "LoadGenerationGapStrategy",
    "LoadSurpriseStrategy",
    "LogisticDirectionStrategy",
    "MajorityVoteMLStrategy",
    "MajorityVoteRuleBasedStrategy",
    "MeanForecastRegressionStrategy",
    "MedianForecastEnsembleStrategy",
    "MedianIndependentStrategy",
    "MondayEffectStrategy",
    "MonthSeasonalStrategy",
    "MonthlyMeanReversionStrategy",
    "MultiSpreadStrategy",
    "NetDemandMomentumStrategy",
    "NetDemandStrategy",
    "NeuralNetStrategy",
    "NLFlowSignalStrategy",
    "NuclearAvailabilityStrategy",
    "NuclearEventStrategy",
    "NuclearGasSubstitutionStrategy",
    "OffshoreWindAnomalyStrategy",
    "PLSRegressionStrategy",
    "PolandSpreadStrategy",
    "PriceMinReversionStrategy",
    "PriceZScoreReversionStrategy",
    "PrunedMLEnsembleStrategy",
    "RadiationSolarStrategy",
    "RadiationRegimeStrategy",
    "QuarterSeasonalStrategy",
    "RandomForestStrategy",
    "RegimeConditionalEnsembleStrategy",
    "RegimeRidgeStrategy",
    "RenewableRegimeStrategy",
    "RenewableRampStrategy",
    "RenewablesPenetrationStrategy",
    "RenewablesSurplusStrategy",
    "RidgeNetDemandStrategy",
    "RidgeRegressionStrategy",
    "SelectiveHighConvictionStrategy",
    "SeasonalRegimeSwitchStrategy",
    "SolarForecastStrategy",
    "SparkSpreadStrategy",
    "SpreadConsensusStrategy",
    "SpreadMomentumStrategy",
    "StackedRidgeMetaStrategy",
    "SVMDirectionStrategy",
    "SupplyDemandBalanceStrategy",
    "TemperatureCurveStrategy",
    "TemperatureExtremeStrategy",
    "TopKEnsembleStrategy",
    "VolatilityBreakoutStrategy",
    "VolatilityRegimeMLStrategy",
    "VolatilityRegimeStrategy",
    "WeekdayWeekendEnsembleStrategy",
    "WeekendMeanReversionStrategy",
    "WeeklyAutocorrelationStrategy",
    "WeeklyCycleStrategy",
    "WeightedVoteMixedStrategy",
    "WindForecastStrategy",
    "WindForecastErrorStrategy",
    "ZScoreMomentumStrategy",
]
