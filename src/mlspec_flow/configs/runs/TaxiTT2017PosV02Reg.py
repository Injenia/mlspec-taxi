from configs.runs.TaxiTripTotalReduced2017FullV01 import TaxiTripTotalReduced2017FullV01Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTT2017PosV02RegRun(TaxiTripTotalReduced2017FullV01Run):
    flow_dir   = "TaxiTripTotalReduced2017FullV01"
    model_name = "20210517V02REG"
    # same as TaxiTripTotalReduced2017FullV01Run but with HPTuning on regularization