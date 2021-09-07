from configs.runs.TaxiTT2017PosCompanyV01 import TaxiTT2017PosCompanyV01Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTT2017PosCompanyV02RegRun(TaxiTT2017PosCompanyV01Run):
    model_name = "20210521V02REG"
    # same as TaxiTT2017PosCompanyV01 but with HPTuning on regularization