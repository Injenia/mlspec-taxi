from configs.runs.TaxiTripTotalReduced2017FullV03 import TaxiTripTotalReduced2017FullV03Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTripTotalReduced2017FullV03OptimizerTestRun(TaxiTripTotalReduced2017FullV03Run):
    model_name = "20210521V01OptimizerTest"
    # This configuration allows to repeat TaxiTripTotalReduced2017FullV02Run 
    # without the hyperparameter tuning, applying the default model size of [32, 16] 