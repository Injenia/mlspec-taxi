from configs.runs.TaxiTS2017MinimalV01 import TaxiTS2017MinimalV01Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTS2017MinimalV02Run(TaxiTS2017MinimalV01Run):
    model_name = "20210521V02"

    def get_default_model_hparams(self):
        batch_size           = 1000000
        epoch_size           = 13946748
        learning_rate        = 0.1
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