from .builder import build_criteria, LOSSES

from .misc import CrossEntropyLoss, SmoothCELoss, DiceLoss, FocalLoss, BinaryFocalLoss
from .lovasz import LovaszLoss
from .weight_regression_loss import RegressionL1Loss