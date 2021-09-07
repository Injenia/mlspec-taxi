from utils.paths import *
from utils.predictor import PredictionModel
from preprocessor.preprocess import PreprocessStep

import apache_beam as beam
import tensorflow as tf


class PredictionModelDoFn(beam.DoFn):
    def __init__(self, flow_dir, model_name):
        self.flow_dir   = flow_dir
        self.model_name = model_name
        
    def setup(self):
        self.model = PredictionModel(self.flow_dir, self.model_name)
        
    def process(self, elements):
        keys, labels, values = elements
        predictions = self.model.predict_batch(values)
        predictions = predictions['predictions'].numpy()[:,0].tolist()
        yield keys, labels, values, predictions
        

class ValidateStep(PreprocessStep):
    fallback_split = 'COVID19'
    def __init__(self, flow_dir, model_name, batch_size):
        assert tf.io.gfile.exists(flow_path(flow_dir)), f"{flow_path(flow_dir)} does not exist"
        query   = load_query(flow_dir)
        columns = load_columns(flow_dir)
        self.flow_dir   = flow_dir
        self.model_name = model_name
        self.columns    = columns
        self.query      = query
        self.batch_size = batch_size
        
        
    def _baseline_value_query(self):
        return f"""
        select AVG({self.columns["label_column"]}) as baseline_prediction
        from ({self.query})
        where {self.columns["split_column"]}="TRAIN"
        """
        
    def _compute_baseline_prediction(self, pipeline):
        source_query = self._baseline_value_query()
        baseline = (
            pipeline
            | f"compute_baseline" >> beam.io.Read(
                beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
        )
        return beam.pvalue.AsSingleton(baseline)
        
    def _prediction_preprocess(self, element):
        keys   = [e[self.columns["key_column"  ]] for e in element]
        labels = [e[self.columns["label_column"]] for e in element]
        values = {k:[e[k] for e in element] 
                  for k in self.columns["categorical_columns"]+self.columns["numeric_columns"]}
        return keys, labels, values
            
    def _prediction_postprocess(self, element, baseline):
        keys, labels, values, predictions = element
        baseline_value = baseline['baseline_prediction']
        rearranged_values = []
        for i in range(len(keys)):
            result = {k:values[k][i] for k in self.columns["categorical_columns"]+self.columns["numeric_columns"]}
            result[self.columns["key_column"  ]] = keys[i]
            result[self.columns["label_column"]] = labels[i]
            result["prediction"] = predictions[i]
            result["baseline"  ] = baseline_value
            yield result
            
    def _add_split(self, step):
        def add_split(element):
            element[self.columns["split_column"]]=self.splits.get(step, self.fallback_split)
            return element
        return add_split
        
    def _predict(self, data, baseline, step):
        x = data
        x = x | f'{step} batch_data'              >> beam.transforms.util.BatchElements(min_batch_size=self.batch_size, max_batch_size=self.batch_size)
        x = x | f'{step} prediction_preprocess'   >> beam.Map(self._prediction_preprocess)
        x = x | f'{step} prediction'              >> beam.ParDo(PredictionModelDoFn(self.flow_dir, self.model_name))
        x = x | f'{step} prediction_postprocess'  >> beam.FlatMap(self._prediction_postprocess, baseline=baseline)
        x = x | f'{step} prediction_finalization' >> beam.Map(self._add_split(step))
        return x
    
    def _write_to_bigquery(self, streams, overwrite_previous):
        table_spec   = bigquery_model_results_table(self.flow_dir, self.model_name)
        table_schema = dict(
            fields = [dict(name=c, type='STRING',  mode='NULLABLE') for c in self.columns["categorical_columns"]
                 ] + [dict(name=c, type='FLOAT64', mode='NULLABLE') for c in self.columns["numeric_columns"]
                 ] + [dict(name=self.columns["key_column"],   type='STRING',  mode='NULLABLE'),
                      dict(name=self.columns["label_column"], type='FLOAT64', mode='NULLABLE'),
                      dict(name=self.columns["split_column"], type='STRING', mode='NULLABLE'),
                      dict(name="prediction",                 type='FLOAT64', mode='NULLABLE'),
                      dict(name="baseline",                   type='FLOAT64', mode='NULLABLE')]
        )
        predictions = (tuple(streams) | 'merge' >> beam.Flatten())
        predictions | 'write_results' >> beam.io.WriteToBigQuery(
            table              = table_spec,
            schema             = table_schema,
            write_disposition  = beam.io.BigQueryDisposition.WRITE_TRUNCATE if overwrite_previous else beam.io.BigQueryDisposition.WRITE_EMPTY,
            create_disposition = beam.io.BigQueryDisposition.CREATE_IF_NEEDED)
        
        
    def predict_data(self, prepare_fn = None, overwrite_previous = False):
        pipeline_options = beam.pipeline.PipelineOptions(
            runner            = 'DataflowRunner',
            project           = config.PROJECT,
            region            = config.REGION,
            job_name          = predict_dataflow_job_name(config.JOB_NAME),
            temp_location     = config.DATAFLOW_STAGING_DIR,
            staging_location  = config.DATAFLOW_TEMP_DIR, 
            save_main_session = True,
            setup_file        = './setup.py'
        )
        with beam.Pipeline('DataflowRunner', options=pipeline_options) as pipeline:
            print(best_model_dir_path(self.flow_dir, self.model_name))
            streams  = []
            baseline = self._compute_baseline_prediction(pipeline)
            for step in ['test', 'covid19']:
                data        = self._read_from_bq(pipeline, step, prepare_fn, self.fallback_split)
                predictions = self._predict(data, baseline, step)
                streams.append(predictions)
            self._write_to_bigquery(streams, overwrite_previous)
    
    