from configs.runs.TaxiTripTotalReduced2017FullV02 import TaxiTripTotalReduced2017FullV02Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTripTotalReduced2017FullV03Run(TaxiTripTotalReduced2017FullV02Run):
    model_name = "20210518"
    # This configuration allows to repeat TaxiTripTotalReduced2017FullV02Run 
    # without the hyperparameter tuning, applying the default model size of [32, 16] 