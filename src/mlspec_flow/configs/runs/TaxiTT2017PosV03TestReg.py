from configs.runs.TaxiTT2017PosV02Reg import TaxiTT2017PosV02RegRun
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTT2017PosV03TestRegRun(TaxiTT2017PosV02RegRun):
    model_name = "20210517V03REGTEST"
    # same as TaxiTripTotalReduced2017FullV01Run but with very high values on regularization
    # this is a sanity check on regularization
    
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
            evaluate_every_epoch = evaluate_every_epoch,
            linear_l1            = 1000000.0,
            linear_l2            = 1000000.0,
            deep_l1              = 1000000.0,
            deep_l2              = 1000000.0
        )