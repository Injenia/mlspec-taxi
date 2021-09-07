from configs.runs.TaxiTripTotalReduced2017V1 import TaxiTripTotalReduced2017V1Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTripTotalReduced2017V2Run(TaxiTripTotalReduced2017V1Run):
    model_name = "20210503V02"
    
    # TRAIN    13946748
    # VALIDATE  2865951
    # TEST      8333301
    # COVID19   1328094
    
    def get_default_model_hparams(self):
        batch_size           = 1000000
        epoch_size           = 13946748
        learning_rate        = 0.01
        num_epochs           = 15
        num_evaluations      = 15
        epoch_steps          = np.ceil(batch_size/epoch_size).astype(int)
        evaluate_every_epoch = int(num_epochs/num_evaluations)
        return dict(
            num_epochs           = num_epochs,
            batch_size           = batch_size, 
            learning_rate        = learning_rate,
            evaluate_every_epoch = evaluate_every_epoch
        )