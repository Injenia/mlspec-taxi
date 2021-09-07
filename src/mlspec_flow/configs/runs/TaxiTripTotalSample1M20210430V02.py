from configs.runs.TaxiTripTotalSample1M20210430V01 import TaxiTripTotalSample1M20210430V01Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np


class TaxiTripTotalSample1M20210430V02Run(TaxiTripTotalSample1M20210430V01Run):
    flow_dir   = "TaxiTripTotalSample1M20210430V0120210430V01"
    model_name = "20210430V02"
    
    def get_default_model_hparams(self):
        batch_size    = 100000
        epoch_size    = 1000000
        learning_rate = 0.01
        hidden_units  = '[256, 64, 16]'
        epoch_steps   = np.ceil(batch_size/epoch_size).astype(int)
        return dict(
            num_epochs         = 100,
            batch_size         = batch_size, 
            learning_rate      = learning_rate,
            hidden_units       = hidden_units,
            evaluate_every_epoch = 10
        )