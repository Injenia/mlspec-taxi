from configs.runs.TaxiTS2017PosV01 import TaxiTS2017PosV01Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTS2017PosV02RegRun(TaxiTS2017PosV01Run):
    model_name = "20210521REG"
    # same as TaxiTS2017PosV01Run but with HPTuning on regularization and more epochs
    
    def get_default_model_hparams(self):
        batch_size           = 1000000
        epoch_size           = 13946748
        learning_rate        = 0.01
        num_epochs           = 20
        num_evaluations      = 20
        epoch_steps          = np.ceil(batch_size/epoch_size).astype(int)
        evaluate_every_epoch = int(num_epochs/num_evaluations)
        return dict(
            num_epochs           = num_epochs,
            batch_size           = batch_size, 
            learning_rate        = learning_rate,
            evaluate_every_epoch = evaluate_every_epoch
        )