from configs.runs.TaxiTripTotal20210429V01 import TaxiTripTotal20210429V01Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTripTotal20210429V03Run(TaxiTripTotal20210429V01Run):
    model_name = "20210429V03"
    
    def get_default_model_hparams(self):
        batch_size    = 76613169
        epoch_size    = 76613169
        epoch_steps   = np.ceil(batch_size/epoch_size).astype(int)
        learning_rate = 0.01
        return dict(
            num_epochs    = 10,
            batch_size    = batch_size, 
            learning_rate = learning_rate,
            decay_steps   = epoch_steps, # decay every epoch
            evaluate_every_epoch = 10
        )