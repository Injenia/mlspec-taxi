import logging
import json
import os

import tensorflow as tf
from tensorflow import data
import tensorflow_transform as tft

from tensorflow_transform.beam import impl
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

from utils.paths import *
from configs.globals import config
import sys, argparse
import numpy as nps

tf_types = {"INT":2, "FLOAT":3}
                    

class TrainWideAndDeepEstimatorStep:
    def __init__(self, 
                 flow_dir,
                 model_name,
                 input_hparams          = dict(),
                 num_epochs             = 1,
                 batch_size             = 500,
                 hidden_units           = json.dumps([32, 16]),
                 extend_feature_columns = True,
                 evaluate_after_sec     = 1,
                 evaluate_every_epoch   = 1,
                 learning_rate          = 0.001,
                 decay_steps            = 10000,
                 decay_rate             = 0.96,
                 linear_l1              = 0.0,
                 linear_l2              = 0.0,
                 deep_l1                = 0.0,
                 deep_l2                = 0.0,
                 cli_params             = []
                ):
        hparams = dict(
            num_epochs             = num_epochs,
            batch_size             = batch_size,
            hidden_units           = hidden_units,
            extend_feature_columns = extend_feature_columns,
            evaluate_after_sec     = evaluate_after_sec,
            evaluate_every_epoch   = evaluate_every_epoch,
            learning_rate          = learning_rate,
            decay_steps            = decay_steps,
            decay_rate             = decay_rate,
            linear_l1              = linear_l1,
            linear_l2              = linear_l2,
            deep_l1                = deep_l1,
            deep_l2                = deep_l2
            
        )
        self.flow_dir              = flow_dir
        self.model_name            = model_name
        self.model_dir             = model_dir(flow_dir, model_name)
        self.export_dir            = export_dir(flow_dir, model_name)
        self.transformed_metadata  = load_metadata(flow_dir)
        self.columns               = load_columns(flow_dir)
        self.transform_output      = load_transform_output(flow_dir)
        self.hparams               = hparams
        self.input_hparams         = input_hparams
        self._parse_cli_and_update_hparams(cli_params)
        
    
    def _parse_cli_and_update_hparams(self, cli_params):
        parser = argparse.ArgumentParser()
        for k,v in self.hparams.items():
            parser.add_argument(f"--{k}", type=type(v), default=v)
        hparams, unknown     = parser.parse_known_args(cli_params)
        hparams.hidden_units = json.loads(hparams.hidden_units)
        self.hparams         = hparams
        
    
        
    def _tfrecords_input_fn(self, files_name_pattern,
                            mode       = tf.estimator.ModeKeys.EVAL,  
                            num_epochs = 1, 
                            batch_size = 500):
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern         = files_name_pattern,
            batch_size           = batch_size,
            features             = self.transform_output.transformed_feature_spec(),
            reader               = tf.data.TFRecordDataset,
            num_epochs           = num_epochs,
            shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False,
            shuffle_buffer_size  = 1+(batch_size*2),
            prefetch_buffer_size = 1
        )
        features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        target = features.pop(self.columns["label_column"])
        return features, target
    
    
    def _create_wide_and_deep_feature_columns(self, extended_feature_columns_fn = None):
        wide_feature_columns = []
        deep_feature_columns = []
        features = self.transformed_metadata.schema.feature
        for feature in features:
            if feature.name == self.columns["label_column"]:
                continue
            if feature.type == tf_types["FLOAT"]:
                deep_feature_columns.append(tf.feature_column.numeric_column(feature.name))
            elif feature.type == tf_types["INT"]:
                if feature.int_domain.is_categorical:
                    wide_feature_columns.append(
                        tf.feature_column.categorical_column_with_identity(
                            feature.name, 
                            num_buckets=feature.int_domain.max+1) )
                else:
                    deep_feature_columns.append(tf.feature_column.numeric_column(feature.name)) 
        if extended_feature_columns_fn is not None:
            extended_wide_feature_columns, extended_deep_feature_columns = extended_feature_columns_fn(self.input_hparams)
            wide_feature_columns = wide_feature_columns + extended_wide_feature_columns
            deep_feature_columns = deep_feature_columns + extended_deep_feature_columns
        logging.debug("Wide columns:")
        logging.debug(wide_feature_columns)
        logging.debug("")
        logging.debug("Deep columns:")
        logging.debug(deep_feature_columns)
        logging.debug("")
        return wide_feature_columns, deep_feature_columns
    
    
    def _create_placeholders(self):
        return dict(
            [(name, tf.keras.Input((), dtype=tf.string, name=name))
             for name in self.columns["categorical_columns"]] +
            [(name, tf.keras.Input((), dtype=tf.float32, name=name))
             for name in self.columns["numeric_columns"]]
        )
    
    
    def _serving_input_receiver_fn(self):
        def serving_input_fn():
            raw_input_features = self._create_placeholders()
            transformed_features = self.transform_output.transform_raw_features(
                raw_input_features, drop_unused_features=True)
            return tf.estimator.export.ServingInputReceiver(
                transformed_features, raw_input_features)
        return serving_input_fn
    
    
    def train_model(self, extended_feature_columns_fn = None, force = False):
        if tf.io.gfile.exists(self.model_dir):
            if force:
                tf.io.gfile.rmtree(self.model_dir)
            else:
                raise Exception(f"{self.model_dir} already exists, please choose another name or set force=True")
        run_config = tf.estimator.RunConfig(
            tf_random_seed = 19830610,
            model_dir      = self.model_dir
        )
        train_data_files = train_data_files_path(self.flow_dir)
        eval_data_files  = eval_data_files_path(self.flow_dir)
        train_spec = tf.estimator.TrainSpec(
          input_fn = lambda: self._tfrecords_input_fn(
            files_name_pattern = train_data_files,
            mode               = tf.estimator.ModeKeys.TRAIN,
            num_epochs         = self.hparams.num_epochs,
            batch_size         = self.hparams.batch_size
          )#,
          # max_steps=hparams.max_steps, # using input_fn epochs since will not perform distributed training
        )
        eval_spec = tf.estimator.EvalSpec(
          input_fn         = lambda: self._tfrecords_input_fn(eval_data_files, batch_size = self.hparams.batch_size),
          throttle_secs    = self.hparams.evaluate_after_sec,
          start_delay_secs = self.hparams.evaluate_after_sec*2,
          steps = None
        )
        wide_feature_columns, deep_feature_columns = self._create_wide_and_deep_feature_columns(extended_feature_columns_fn)
        logging.info(f"model will be saved to {run_config.model_dir}")
        _="""
        linear_optimizer_fn = lambda: tf.keras.optimizers.Ftrl(
                        learning_rate = tf.compat.v1.train.exponential_decay(
                            learning_rate = self.hparams.learning_rate,
                            global_step   = tf.compat.v1.train.get_global_step(),
                            decay_steps   = self.hparams.decay_steps,
                            decay_rate    = self.hparams.decay_rate))
        deep_optimizer_fn   = lambda: tf.keras.optimizers.Adagrad(
                        learning_rate = tf.compat.v1.train.exponential_decay(
                            learning_rate = self.hparams.learning_rate,
                            global_step   = tf.compat.v1.train.get_global_step(),
                            decay_steps   = self.hparams.decay_steps,
                            decay_rate    = self.hparams.decay_rate))
        """
        linear_optimizer_fn = lambda: tf.keras.optimizers.Ftrl(
            learning_rate              = self.hparams.learning_rate,
            l1_regularization_strength = self.hparams.linear_l1,
            l2_regularization_strength = self.hparams.linear_l2
        )
        # can't apply regularization to the DNN part as explained here: https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier
        # due to a bug which as of today (21/05/2021) is still not resolved: https://github.com/tensorflow/tensorflow/issues/46342
        #deep_optimizer_fn   = tf.compat.v1.train.ProximalAdagradOptimizer(
        #    learning_rate              = self.hparams.learning_rate,
        #    l1_regularization_strength = self.hparams.deep_l1,
        #    l2_regularization_strength = self.hparams.deep_l2
        #)
        
        # this is the other optimizer in the tutorial
        #deep_optimizer_fn   = lambda: tf.keras.optimizers.Adagrad(learning_rate = self.hparams.learning_rate)
        
        # this is the only keras optimizer having l1 and l2 regularizations
        deep_optimizer_fn = lambda: tf.keras.optimizers.Ftrl( 
            learning_rate              = self.hparams.learning_rate,
            l1_regularization_strength = self.hparams.deep_l1,
            l2_regularization_strength = self.hparams.deep_l2
        )
        
        estimator = tf.estimator.DNNLinearCombinedRegressor(
                        linear_feature_columns = wide_feature_columns,
                        dnn_feature_columns    = deep_feature_columns,
                        dnn_hidden_units       = self.hparams.hidden_units,
                        config                 = run_config,
                        model_dir              = run_config.model_dir,
                        linear_optimizer       = linear_optimizer_fn,
                        dnn_optimizer          = deep_optimizer_fn
                        )
        time_start = datetime.utcnow() 
        logging.info("")
        logging.info("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
        logging.info(".......................................") 
        #tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
        for epoch in range(0,self.hparams.num_epochs,self.hparams.evaluate_every_epoch):
            logging.info(f"------------------------------------------- TRAINING FROM EPOCH {epoch} -------------------------------------------")
            remaining_epochs = self.hparams.num_epochs - epoch
            estimator.train(input_fn = lambda: self._tfrecords_input_fn(
                files_name_pattern = train_data_files,
                mode               = tf.estimator.ModeKeys.TRAIN,
                num_epochs         = min(self.hparams.evaluate_every_epoch, remaining_epochs),
                batch_size         = self.hparams.batch_size
              ))
            logging.info("------------------------------------------- EVALUATING EPOCH {} -------------------------------------------".format(epoch+min(self.hparams.evaluate_every_epoch, remaining_epochs)))
            estimator.evaluate(input_fn = lambda: self._tfrecords_input_fn(eval_data_files, batch_size = self.hparams.batch_size))
        
        time_end = datetime.utcnow() 
        logging.info(".......................................")
        logging.info("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
        logging.info("")
        time_elapsed = time_end - time_start
        logging.info("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
        estimator.export_saved_model(
            export_dir_base=self.export_dir,
            serving_input_receiver_fn=self._serving_input_receiver_fn()
        )