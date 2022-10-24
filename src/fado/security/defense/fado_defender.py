from importlib import import_module
import logging
from typing import List, Tuple, Dict, Any, Callable
from fedml.core.security.defense.RFA_defense import RFADefense
from fedml.core.security.defense.coordinate_wise_trimmed_mean_defense import CoordinateWiseTrimmedMeanDefense
from fedml.core.security.defense.crfl_defense import CRFLDefense
from fedml.core.security.defense.three_sigma_defense import ThreeSigmaDefense
from fedml.core.security.defense.three_sigma_geomedian_defense import ThreeSigmaGeoMedianDefense
from fedml.core.common.ml_engine_backend import MLEngineBackend
from fedml.core.security.defense.cclip_defense import CClipDefense
from fedml.core.security.defense.foolsgold_defense import FoolsGoldDefense
from fedml.core.security.defense.geometric_median_defense import GeometricMedianDefense
from .factory.krum import KrumDefense
from fedml.core.security.defense.robust_learning_rate_defense import RobustLearningRateDefense
from fedml.core.security.defense.slsgd_defense import SLSGDDefense
from fedml.core.security.defense.weak_dp_defense import WeakDPDefense
from fedml.core.security.defense.norm_diff_clipping_defense import NormDiffClippingDefense
from fedml.core.security.constants import (
    DEFENSE_NORM_DIFF_CLIPPING,
    DEFENSE_ROBUST_LEARNING_RATE,
    DEFENSE_KRUM,
    DEFENSE_SLSGD,
    DEFENSE_GEO_MEDIAN,
    DEFENSE_CCLIP,
    DEFENSE_WEAK_DP,
    DEFENSE_RFA,
    DEFENSE_FOOLSGOLD,
    DEFENSE_THREESIGMA,
    DEFENSE_CRFL,
    DEFENSE_MULTIKRUM,
    DEFENSE_TRIMMED_MEAN,
    DEFENSE_THREESIGMA_GEOMEDIAN,
)

logger = logging.getLogger("fado")


class FadoDefender:
    _defender_instance = None

    @staticmethod
    def get_instance():
        if FadoDefender._defender_instance is None:
            FadoDefender._defender_instance = FadoDefender()

        return FadoDefender._defender_instance

    def __init__(self):
        self.is_enabled = False
        self.defender = None

    def init(self, args):
        if hasattr(args, "defense_spec") and args.defense_spec:
            self.args = args
            self.defender = None

            if args.rank != 0:  # do not initialize defense for client
                return

            if isinstance(args.defense_spec, str):
                defense_pkg, defense_module, defense_class = args.defense_class.split('.')
                self.defender = getattr(import_module(f'{defense_pkg}.{defense_module}'), f'{defense_class}')(args)
            else:
                self.defender = args.defense_spec(args)

            logger.info(f"Initializing defender! {self.defender}")

    def is_defense_enabled(self):
        return self.defender is not None

    def defend(
            self,
            raw_client_grad_list: List[Tuple[float, Dict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
    ):

        raise Exception("Why was this executed?!")

        """
        if self.defender is None:
            raise Exception("defender is not initialized!")
        return self.defender.run(
            raw_client_grad_list, base_aggregation_func, extra_auxiliary_info
        )
        """

    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, Dict]],
            extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")

        if callable(getattr(self.defender, "defend_before_aggregation", None)):
            return self.defender.defend_before_aggregation(
                raw_client_grad_list, extra_auxiliary_info
            )
        return raw_client_grad_list

    def defend_on_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, Dict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")

        if callable(getattr(self.defender, "defend_on_aggregation", None)):
            return self.defender.defend_on_aggregation(
                raw_client_grad_list, base_aggregation_func, extra_auxiliary_info
            )

        return base_aggregation_func(args=self.args, raw_grad_list=raw_client_grad_list)

    def defend_after_aggregation(self, global_model):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        if callable(getattr(self.defender, "defender_after_aggregation", None)):
            return self.defender.defend_after_aggregation(global_model)
        return global_model
