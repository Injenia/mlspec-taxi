from datetime import datetime
import logging
import json
import os

import tensorflow as tf
from tensorflow import data
import apache_beam as beam
import tensorflow_transform as tft

from tensorflow_transform.beam import impl
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

from utils.paths import *
from configs.globals import config


class PreprocessStep:
    splits = dict(train="TRAIN", eval="VALIDATE", test="TEST")
    
    def __init__(self, flow_dir):
        assert tf.io.gfile.exists(flow_path(flow_dir)), f"{flow_path(flow_dir)} does not exist"
        query   = load_query(flow_dir)
        columns = load_columns(flow_dir)
        self.flow_dir = flow_dir
        self.columns  = columns
        self.query    = query
        
        
    def _create_raw_metadata(self):
        raw_data_schema = {}
    
        # key feature schema
        raw_data_schema[self.columns["key_column"]]= dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation())

        # target feature schema
        raw_data_schema[self.columns["label_column"]]= dataset_schema.ColumnSchema(
            tf.float32, [], dataset_schema.FixedColumnRepresentation())

        # categorical features schema
        raw_data_schema.update({ column_name : dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation())
                                for column_name in self.columns["categorical_columns"]})

        # numerical features schema
        raw_data_schema.update({ column_name : dataset_schema.ColumnSchema(
            tf.float32, [], dataset_schema.FixedColumnRepresentation())
                                for column_name in self.columns["numeric_columns"]})

          # create dataset_metadata given raw_schema
        raw_metadata = dataset_metadata.DatasetMetadata(
            dataset_schema.Schema(raw_data_schema))

        return raw_metadata
        
    
    def _get_source_query(self, split):
        return f"""
        select * except({self.columns["split_column"]})
        from ({self.query})
        where {self.columns["split_column"]}="{split}"
        """
    
    def _read_from_bq(self, pipeline, step, prepare_fn=None, fallback_split="GOLDEN"):
        split        = self.splits.get(step, fallback_split)
        source_query = self._get_source_query(split)
        raw_data = (
            pipeline
            | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
                beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
        )
        if prepare_fn is not None:
            raw_data = (
                raw_data
                | '{} - Clean up Data'.format(step) >> beam.Map(prepare_fn)
            )
        return raw_data
        
    
    def _read_dataset_from_bq(self, pipeline, step, prepare_fn=None, fallback_split="GOLDEN"):
        split        = self.splits.get(step, fallback_split)
        source_query = self._get_source_query(split)
        raw_data = self._read_from_bq(pipeline, step, prepare_fn, fallback_split)
        raw_metadata = self._create_raw_metadata()
        raw_dataset = (raw_data, raw_metadata)
        return raw_dataset
    
        
    def _analyze_and_transform(self, raw_dataset, step, preprocess_fn):
        transformed_dataset, transform_fn = (
            raw_dataset 
            | '{} - Analyze & Transform'.format(step) >> impl.AnalyzeAndTransformDataset(preprocess_fn)
        )
        return transformed_dataset, transform_fn
    
    
    def _transform(self, raw_dataset, transform_fn, step):
        transformed_dataset = (
            (raw_dataset, transform_fn) 
            | '{} - Transform'.format(step) >> impl.TransformDataset()
        )
        return transformed_dataset
    
    
    def _write_tfrecords(self, transformed_dataset, location, step):
        transformed_data, transformed_metadata = transformed_dataset
        (
            transformed_data 
            | '{} - Write Transformed Data'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=os.path.join(location,'{}'.format(step)),
                file_name_suffix=".tfrecords",
                coder=example_proto_coder.ExampleProtoCoder(transformed_metadata.schema))
        )
        
        
    def _write_transform_artefacts(self, transform_fn, location):
        (
            transform_fn 
            | 'Write Transform Artefacts' >> transform_fn_io.WriteTransformFn(location)
        )
    
    
    def _write_text(self, dataset, location, step):
        data, _ = dataset
        (
            data 
            | '{} - WriteData'.format(step) >> beam.io.WriteToText(
                file_path_prefix=os.path.join(location,'{}'.format(step)),
                file_name_suffix=".txt")
        )
    
    
    def preprocess_data(self, preprocess_fn, prepare_fn = None, debug = False, force = False):
        preprocess_path = preprocessing_path(self.flow_dir)
        if tf.io.gfile.exists(preprocess_path):
            if force:
                tf.io.gfile.rmtree(preprocess_path)
            else:
                raise Exception(f"{preprocess_path} already exists, please choose another name or set force=True")
        
        pipeline_options = beam.pipeline.PipelineOptions(
            runner            = 'DataflowRunner',
            project           = config.PROJECT,
            region            = config.REGION,
            job_name          = preprocess_dataflow_job_name(self.flow_dir),
            temp_location     = config.DATAFLOW_STAGING_DIR,
            staging_location  = config.DATAFLOW_TEMP_DIR, 
            save_main_session = True,
            setup_file        = './setup.py'
        )
        transformed_data_location   = transformed_data_dir(self.flow_dir)
        transform_artefact_location = transform_artefacts_dir(self.flow_dir)
        temp_dir                    = temporary_dir(self.flow_dir)
        raw_metadata                = self._create_raw_metadata()
        with beam.Pipeline('DataflowRunner', options=pipeline_options) as pipeline:
            with impl.Context(temp_dir):
                step = 'train'
                raw_train_dataset = self._read_dataset_from_bq(pipeline, step, prepare_fn)
                transformed_train_dataset, transform_fn = self._analyze_and_transform(raw_train_dataset, step, preprocess_fn)
                self._write_tfrecords(transformed_train_dataset, transformed_data_location, step)
                self._write_transform_artefacts(transform_fn, transform_artefact_location)
                step = 'eval'
                raw_eval_dataset = self._read_dataset_from_bq(pipeline, step, prepare_fn)
                transformed_eval_dataset = self._transform(raw_eval_dataset, transform_fn, step)
                self._write_tfrecords(transformed_eval_dataset, transformed_data_location, step)
            if debug:
                step = 'debug'
                self._write_text(transformed_train_dataset, transformed_data_location, step)