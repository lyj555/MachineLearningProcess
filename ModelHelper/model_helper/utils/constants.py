# -*- coding: utf-8 -*-

from enum import Enum


class JobType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelStage(Enum):
    TRAIN = "train"
    PREDICT = "predict"

