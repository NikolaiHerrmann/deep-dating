
from enum import Enum


class SetType(Enum):
    TEST = "test"
    VAL = "val"
    TRAIN = "train"


class DatasetName(Enum):
    MPS = "mps"
    CLAMM = "clamm"
    SCRIBBLE = "scribble"
    DIBCO = "dibco"

    def __str__(self):
        return self.name