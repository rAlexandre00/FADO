import copy
import logging
from typing import List, Tuple, Dict
from fedml.ml.aggregator.agg_operator import FedMLAggOperator
from fedml.core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import torch
from torch import nn
from fedml.core.alg_frame.server_aggregator import ServerAggregator
import numpy as np

from defense.fado_defender import FadoDefender

logger = logging.getLogger(__name__)

class FadoServerAggregator(ServerAggregator):
    def __init__(self, model, args):
        super().__init__(model, args)
        FadoDefender.get_instance().init(args)
        self.cpu_transfer = False if not hasattr(self.args, "cpu_transfer") else self.args.cpu_transfer

    def get_model_params(self):
        if self.cpu_transfer:
            return self.model.cpu().state_dict()
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def _test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        if args.dataset == "stackoverflow_lr":
            criterion = nn.BCELoss(reduction="sum").to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > 0.5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics["test_precision"] += precision.sum().item()
                    metrics["test_recall"] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        return metrics

    def test(self, test_data, device, args):
        # test data
        test_num_samples = []
        test_tot_corrects = []
        test_losses = []

        metrics = self._test(test_data, device, args)

        test_tot_correct, test_num_sample, test_loss = (
            metrics["test_correct"],
            metrics["test_total"],
            metrics["test_loss"],
        )
        test_tot_corrects.append(copy.deepcopy(test_tot_correct))
        test_num_samples.append(copy.deepcopy(test_num_sample))
        test_losses.append(copy.deepcopy(test_loss))

        # test on test dataset
        test_acc = sum(test_tot_corrects) / sum(test_num_samples)
        test_loss = sum(test_losses) / sum(test_num_samples)
        
        logging.trace(f"Round: {args.round_idx}")
        logging.trace(f"\tTest/Acc: {test_acc}")
        logging.trace(f"\tTest/Loss: {test_loss}")

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        train_num_samples = []
        train_tot_corrects = []
        train_losses = []
        for client_idx in range(self.args.client_num_in_total):
            # train data
            metrics = self._test(train_data_local_dict[client_idx], device, args)
            train_tot_correct, train_num_sample, train_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            train_num_samples.append(copy.deepcopy(train_num_sample))
            train_losses.append(copy.deepcopy(train_loss))
            # logging.info("testing client_idx = {}".format(client_idx))
            # logger.trace("testing client_idx = {}".format(client_idx))

        # test on training dataset
        train_acc = sum(train_tot_corrects) / sum(train_num_samples)
        train_loss = sum(train_losses) / sum(train_num_samples)

        logging.trace(f"Round: {args.round_idx}")
        logging.trace(f"\tTrain/Acc: {train_acc}")
        logging.trace(f"\tTrain/Loss: {train_loss}")

        return True

    # OVERRIDING THE FOLLOWING METHODS

    def on_before_aggregation(
        self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]
    ) -> List[Tuple[float, Dict]]:
        if FadoDefender.get_instance().is_defense_enabled():
                raw_client_model_or_grad_list = FadoDefender.get_instance().defend_before_aggregation(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    extra_auxiliary_info=self.get_model_params(),
                )
        return raw_client_model_or_grad_list

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]) -> Dict:
        if FadoDefender.get_instance().is_defense_enabled():
            return FedMLDefender.get_instance().defend_on_aggregation(
                raw_client_grad_list=raw_client_model_or_grad_list,
                base_aggregation_func=FedMLAggOperator.agg,
                extra_auxiliary_info=self.get_model_params(),
            )
        return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)


    def on_after_aggregation(self, aggregated_model_or_grad: Dict) -> Dict:
        if FedMLDifferentialPrivacy.get_instance().is_global_dp_enabled():
            logging.info("-----add central DP noise ----")
            aggregated_model_or_grad = FedMLDifferentialPrivacy.get_instance().add_global_noise(
                aggregated_model_or_grad
            )
        if FadoDefender.get_instance().is_defense_enabled():
            aggregated_model_or_grad = FadoDefender.get_instance().defend_after_aggregation(aggregated_model_or_grad)
        return aggregated_model_or_grad
        