import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillLoss:
    def __init__(self, criterion: nn.Module, teacher_model: nn.Module,
                 alpha: float, tau: float, loss_type: str):
        self.criterion = criterion
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.tau = tau

        # Check the loss type
        assert loss_type in ("soft", "hard", "none"), "Not supported type."
        self.loss_type = loss_type

    def __call__(self, inputs, student_outputs, labels, loss_type):
        outputs_cls, outputs_dis = student_outputs

        # Compute the base criterion with ground truth hard labels
        base_loss = self.criterion(outputs_cls, labels)

        # Compute the teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        # Set initial to 0.0
        distillation_loss = 0.0

        # Process different types of loss function
        if self.loss_type == "soft":
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_dis / self.tau, dim=-1),
                F.log_softmax(teacher_outputs / self.tau, dim=-1),
                log_target=True
            ) * (self.tau**2)
        elif self.loss_type == "hard":
            distillation_loss = F.cross_entropy(
                outputs_dis,
                teacher_outputs.argmax(dim=-1)
            )

        # Mix up all type of loss
        total_loss = base_loss * (1 - self.alpha) + distillation_loss * (1 - self.alpha)
