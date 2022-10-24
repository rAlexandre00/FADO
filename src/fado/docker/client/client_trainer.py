from typing import List
from fedml.core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import torch
from torch import nn

from fedml.core.alg_frame.client_trainer import ClientTrainer
import logging

from fado.security.attack import FadoAttacker

logger = logging.getLogger(__name__)

class FadoClientTrainer(ClientTrainer):

    def __init__(self, model, args):
        """Initialize class

        Args:
            model: Training model
            args: Runtime arguments
        """
        super().__init__(model, args)
        # Initialize attacker class
        FadoAttacker.get_instance().init(args)

    
    def on_before_local_training(self, train_data, device, args):
        """Method called before the main training process

        If a data attack is specified, replace the train data using the
        attack_data method defined in the attack class

        Args:
            train_data: Full training dataset
        """
        if FadoAttacker.get_instance().is_data_attack():
            # Attack data and replace array content
            train_data[:] = FadoAttacker.get_instance().attack_data(train_data)

    def on_after_local_training(self, train_data, device, args):
        """Method called after the main training process

        If a model attack is specified, submit the current model parameters
        to the attack module, which returned model parameters will be assigned
        to the current ones.

        If differential privacy is enabled, local noise will be added. This noise is
        added by FedML differential privacy module.

        Args:
            The arguments in the method are not useful.

        """
        if FadoAttacker.get_instance().is_model_attack():
            self.set_model_params(FadoAttacker.get_instance().attack_model(self.get_model_params()))

        if FedMLDifferentialPrivacy.get_instance().is_local_dp_enabled():
            model_params_with_dp_noise = FedMLDifferentialPrivacy.get_instance().add_local_noise(self.get_model_params())
            self.set_model_params(model_params_with_dp_noise)
        
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        """Trains the model using train_data

        Args:
            train_data: Full training dataset
            device: Device that will be used to train, this is handled by FedML
            args: Runtime arguments, useful for customizing the training process
        """
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
