import tensorflow as tf
from utils.paths import *

class PredictionModel:
    def __init__(self, flow_dir, model_name, signature="predict"):
        self.model        = tf.saved_model.load(saved_model_dir(flow_dir, model_name))
        self.predictor_fn = self.model.signatures[signature]
        
    def signature(self):
        return self.predictor_fn.structured_input_signature
    
    def predict(self, **kwargs):
        pred_input = {k:tf.constant([v]) for k,v in kwargs.items()}
        return self.predictor_fn(**pred_input)
    
    def predict_batch(self, batch):
        pred_input = {k:tf.constant(v) for k,v in batch.items()}
        return self.predictor_fn(**pred_input)