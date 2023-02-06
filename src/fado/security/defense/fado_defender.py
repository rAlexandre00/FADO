from importlib import import_module
import logging
from typing import List, Tuple, Dict, Any, Callable

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
