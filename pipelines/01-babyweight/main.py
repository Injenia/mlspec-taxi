PROJECT = 'mlteam-ml-specialization-2021' # change to your project_Id
BUCKET = 'mlteam-ml-specialization-2021-taxi' # change to your bucket name
REGION = 'europe-west1' # change to your region
ROOT_DIR = 'babyweight_tft' # directory where the output is stored locally or on GCS
STAGING_DIR='tmp'

RUN_LOCAL = False # if True, the DirectRunner is used, else DataflowRunner
DATA_SIZE = 10000 # number of records to be retrieved from BigQuery


import os

import tensorflow as tf
import apache_beam as beam
import tensorflow_transform as tft

from tensorflow_transform.beam import impl
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

import time, datetime
def get_current_timestamp():
    return datetime.datetime.utcnow().strftime('%y%m%d%H%M%S')

current_timestamp = get_current_timestamp()


OUTPUT_DIR = ROOT_DIR if RUN_LOCAL==True else "gs://{}/{}".format(BUCKET,ROOT_DIR)
TRANSFORM_ARTEFACTS_DIR = os.path.join(OUTPUT_DIR,'transform')
TRANSFORMED_DATA_DIR = os.path.join(OUTPUT_DIR,'transformed')
TEMP_DIR = os.path.join(OUTPUT_DIR, 'tmp')
DATAFLOW_STAGING_DIR="gs://{}/{}/staging".format(BUCKET,STAGING_DIR)
DATAFLOW_TEMP_DIR="gs://{}/{}/temp".format(BUCKET,STAGING_DIR)

runner = 'DirectRunner' if RUN_LOCAL == True else 'DataflowRunner'

job_name = 'preprocess-babweight-data-tft-{}'.format(current_timestamp)

#args = {
#    
#    'job_name': job_name,
#    'runner': runner,
#    'data_size': DATA_SIZE,
#    'transformed_data_location':  TRANSFORMED_DATA_DIR,
#    'transform_artefact_location':  TRANSFORM_ARTEFACTS_DIR,
#    'temporary_dir': TEMP_DIR,
#    'debug':False,
#    
#    'project': PROJECT,
#    'region': REGION,
#    'staging_location': os.path.join(OUTPUT_DIR, 'staging'),
#    'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
#    'worker_machine_type': 'n1-standard-1',
#    #'requirements_file': 'requirements.txt',
#}

args = dict(
    runner=runner,
    project=PROJECT,
    region=REGION,
    job_name=job_name,
    temp_location=DATAFLOW_STAGING_DIR,
    staging_location=DATAFLOW_TEMP_DIR,
    save_main_session=True,
    setup_file='./setup.py',
    
    data_size=DATA_SIZE,
    transformed_data_location=TRANSFORMED_DATA_DIR,
    transform_artefact_location=TRANSFORM_ARTEFACTS_DIR,
    temporary_dir=TEMP_DIR,
    debug=False
    
)



CATEGORICAL_FEATURE_NAMES = ['is_male', 'mother_race']
NUMERIC_FEATURE_NAMES = [
    'mother_age', 
    'plurality', 
    'gestation_weeks'
]
TARGET_FEATURE_NAME = 'weight_pounds'
KEY_COLUMN = 'key'

def create_raw_metadata():  
    
    raw_data_schema = {}
    
    # key feature schema
    raw_data_schema[KEY_COLUMN]= dataset_schema.ColumnSchema(
        tf.float32, [], dataset_schema.FixedColumnRepresentation())
    
    # target feature schema
    raw_data_schema[TARGET_FEATURE_NAME]= dataset_schema.ColumnSchema(
        tf.float32, [], dataset_schema.FixedColumnRepresentation())
    
    # categorical features schema
    raw_data_schema.update({ column_name : dataset_schema.ColumnSchema(
        tf.string, [], dataset_schema.FixedColumnRepresentation())
                            for column_name in CATEGORICAL_FEATURE_NAMES})
    
    # numerical features schema
    raw_data_schema.update({ column_name : dataset_schema.ColumnSchema(
        tf.float32, [], dataset_schema.FixedColumnRepresentation())
                            for column_name in NUMERIC_FEATURE_NAMES})
    
      # create dataset_metadata given raw_schema
    raw_metadata = dataset_metadata.DatasetMetadata(
        dataset_schema.Schema(raw_data_schema))
    
    return raw_metadata

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(create_raw_metadata().schema)




def get_source_query(step, data_size):
    
    train_size = data_size * 0.7
    eval_size = data_size * 0.3
    
    query = """
    SELECT
      ROUND(weight_pounds,1) AS weight_pounds,
      is_male,
      mother_age,
      mother_race,
      plurality,
      gestation_weeks,
      FARM_FINGERPRINT( 
        CONCAT(
          COALESCE(CAST(weight_pounds AS STRING), 'NA'),
          COALESCE(CAST(is_male AS STRING),'NA'),
          COALESCE(CAST(mother_age AS STRING),'NA'),
          COALESCE(CAST(mother_race AS STRING),'NA'),
          COALESCE(CAST(plurality AS STRING), 'NA'),
          COALESCE(CAST(gestation_weeks AS STRING),'NA')
          )
        ) AS key
        FROM
          publicdata.samples.natality
        WHERE year > 2000
        AND weight_pounds > 0
        AND mother_age > 0
        AND plurality > 0
        AND gestation_weeks > 0
        AND month > 0
    """
    
    if step == 'train':
        source_query = 'SELECT * FROM ({}) WHERE MOD(key, 100) < 70 LIMIT {}'.format(query,int(train_size))
    else:
        source_query = 'SELECT * FROM ({}) WHERE MOD(key, 100) >= 70 LIMIT {}'.format(query,int(eval_size))
    
    return source_query





def prep_bq_row(bq_row):

    # modify opaque numeric race code into human-readable data
    races = dict(zip([1,2,3,4,5,6,7,18,28,39,48],
                     ['White', 'Black', 'American Indian', 'Chinese', 
                      'Japanese', 'Hawaiian', 'Filipino',
                      'Asian Indian', 'Korean', 'Samaon', 'Vietnamese']))
    result = {} 
    
    for feature_name in bq_row.keys():
        result[feature_name] = str(bq_row[feature_name]) if feature_name in CATEGORICAL_FEATURE_NAMES else bq_row[feature_name]

    if 'mother_race' in bq_row and bq_row['mother_race'] in races:
        result['mother_race'] = races[bq_row['mother_race']]
    else:
        result['mother_race'] = 'Unknown'

    return result




def read_from_bq(pipeline, step, data_size):
    
    source_query = get_source_query(step, data_size)
    raw_data = (
        pipeline
        | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
            beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
        | '{} - Clean up Data'.format(step) >> beam.Map(prep_bq_row)
    )
    
    raw_metadata = create_raw_metadata()
    raw_dataset = (raw_data, raw_metadata)
    return raw_dataset




def preprocess_fn(input_features):
    
    output_features = {}

    # target feature
    output_features['weight_pounds'] = input_features['weight_pounds']

    # normalisation
    output_features['mother_age_normalized'] = tft.scale_to_z_score(input_features['mother_age'])
    # output_features['gestation_weeks_normalized'] =  tft.scale_to_0_1(input_features['gestation_weeks'])
    
    # bucktisation based on quantiles
    output_features['mother_age_bucketized'] = tft.bucketize(input_features['mother_age'], num_buckets=5)
    
    # you can compute new features based on custom formulas
    output_features['mother_age_log'] = tf.math.log(input_features['mother_age'])
    
    # or create flags/indicators
    is_multiple = tf.as_string(input_features['plurality'] > tf.constant(1.0))
    
    # convert categorical features to indexed vocab
    output_features['mother_race_index'] = tft.compute_and_apply_vocabulary(input_features['mother_race'], vocab_filename='mother_race')
    output_features['is_male_index'] = tft.compute_and_apply_vocabulary(input_features['is_male'], vocab_filename='is_male')
    output_features['is_multiple_index'] = tft.compute_and_apply_vocabulary(is_multiple, vocab_filename='is_multiple')
    
    return output_features





def analyze_and_transform(raw_dataset, step):
    
    transformed_dataset, transform_fn = (
        raw_dataset 
        | '{} - Analyze & Transform'.format(step) >> impl.AnalyzeAndTransformDataset(preprocess_fn)
    )
    
    return transformed_dataset, transform_fn




def transform(raw_dataset, transform_fn, step):
    
    transformed_dataset = (
        (raw_dataset, transform_fn) 
        | '{} - Transform'.format(step) >> impl.TransformDataset()
    )
    
    return transformed_dataset




def write_tfrecords(transformed_dataset, location, step):
    
    transformed_data, transformed_metadata = transformed_dataset
    (
        transformed_data 
        | '{} - Write Transformed Data'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix=os.path.join(location,'{}'.format(step)),
            file_name_suffix=".tfrecords",
            coder=example_proto_coder.ExampleProtoCoder(transformed_metadata.schema))
    )
    
    
    
def write_text(dataset, location, step):
    
    data, _ = dataset
    (
        data 
        | '{} - WriteData'.format(step) >> beam.io.WriteToText(
            file_path_prefix=os.path.join(location,'{}'.format(step)),
            file_name_suffix=".txt")
    )
    
    
def write_transform_artefacts(transform_fn, location):
    
    (
        transform_fn 
        | 'Write Transform Artefacts' >> transform_fn_io.WriteTransformFn(location)
    )
    
    

def run_transformation_pipeline(args):
    
    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **args)
    
    runner = args['runner']
    data_size = args['data_size']
    transformed_data_location = args['transformed_data_location']
    transform_artefact_location = args['transform_artefact_location']
    temporary_dir = args['temporary_dir']
    debug = args['debug']
    
    print("Sample data size: {}".format(data_size))
    print("Sink transformed data files location: {}".format(transformed_data_location))
    print("Sink transform artefact location: {}".format(transform_artefact_location))
    print("Temporary directory: {}".format(temporary_dir))
    print("Runner: {}".format(runner))
    print("Debug enabled: {}".format(debug))

    with beam.Pipeline(runner, options=pipeline_options) as pipeline:
        with impl.Context(temporary_dir):
            
            # Preprocess train data
            step = 'train'
            # Read raw train data from BQ
            raw_train_dataset = read_from_bq(pipeline, step, data_size)
            # Analyze and transform raw_train_dataset 
            transformed_train_dataset, transform_fn = analyze_and_transform(raw_train_dataset, step)
            # Write transformed train data to sink as tfrecords
            write_tfrecords(transformed_train_dataset, transformed_data_location, step)
            
            # Preprocess evaluation data
            step = 'eval'
            # Read raw eval data from BQ
            raw_eval_dataset = read_from_bq(pipeline, step, data_size)
            # Transform eval data based on produced transform_fn
            transformed_eval_dataset = transform(raw_eval_dataset, transform_fn, step)
            # Write transformed eval data to sink as tfrecords
            write_tfrecords(transformed_eval_dataset, transformed_data_location, step)
            
            # Write transformation artefacts 
            write_transform_artefacts(transform_fn, transform_artefact_location)

            # (Optional) for debugging, write transformed data as text 
            step = 'debug'
            # Wwrite transformed train data as text if debug enabled
            if debug == True:
                write_text(transformed_train_dataset, transformed_data_location, step)

                
                
                





import logging

try: 
    tf.gfile.DeleteRecursively(TRANSFORMED_DATA_DIR)
    tf.gfile.DeleteRecursively(TRANSFORM_ARTEFACTS_DIR)
    tf.gfile.DeleteRecursively(TEMP_DIR)
    print('previous transformation files deleted!')
except:
    pass

tf.get_logger().setLevel(logging.ERROR)
print('Launching {} job {} ... hang on'.format(runner, job_name))
print("")
run_transformation_pipeline(args)
print("Done!")