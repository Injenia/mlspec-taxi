from configs.runs.TaxiTT2017PosTimeCompanyV01 import TaxiTT2017PosTimeCompanyV01Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTT2017PosTimeCompanyV02RegRun(TaxiTT2017PosTimeCompanyV01Run):
    model_name = "20210521V02REG"
    # same as TaxiTT2017PosTimeCompanyV01Run but with HPTuning on regularization
