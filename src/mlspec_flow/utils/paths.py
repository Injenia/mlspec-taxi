from configs.globals import config
from datetime import datetime
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import metadata_io
import json
import os


def current_time():
    return datetime.utcnow().strftime('%y%m%d%H%M%S')

def flow_path(flow_dir):
    return "gs://{}/{}/{}".format(config.BUCKET, config.ROOT_DIR, flow_dir)

def query_path(flow_dir):
    return os.path.join(flow_path(flow_dir), "init", "query.sql")

def columns_path(flow_dir):
    return os.path.join(flow_path(flow_dir), "init", "columns.json")

def preprocessing_path(flow_dir):
    return os.path.join(flow_path(flow_dir), "preprocessing")

def transform_artefacts_dir(flow_dir):
    return os.path.join(preprocessing_path(flow_dir), "transform")

def transformed_data_dir(flow_dir):
    return os.path.join(preprocessing_path(flow_dir), "transformed")

def temporary_dir(flow_dir):
    return os.path.join(preprocessing_path(flow_dir), "tmp")

def models_dir(flow_dir):
    return os.path.join(flow_path(flow_dir), "models")

def model_runs_dir(flow_dir, model_name):
    return os.path.join(models_dir(flow_dir), model_name)

def model_dir(flow_dir, model_name):
    on_cloud = 'TF_CONFIG' in os.environ
    trial    = '1'
    if on_cloud:
        is_master = json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('type', '') == "master"
        trial     = str(json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '1'))
    return os.path.join(model_runs_dir(flow_dir, model_name), trial)
    
def export_dir(flow_dir, model_name):
    return os.path.join(model_dir(flow_dir, model_name), 'export')

def transformed_metadata_dir(flow_dir):
    return os.path.join(transform_artefacts_dir(flow_dir), "transformed_metadata")

def dataflow_job_name(flow_dir):
    return "{}-{}".format(flow_dir.lower().replace("_",""), current_time())

def preprocess_dataflow_job_name(flow_dir):
    return "preprocess-{}".format(dataflow_job_name(flow_dir))

def predict_dataflow_job_name(flow_dir):
    return "predict-{}".format(dataflow_job_name(flow_dir))

def load_query(flow_dir):
    return tf.io.gfile.GFile(query_path(flow_dir), "r").read()

def load_columns(flow_dir):
    return json.load(tf.io.gfile.GFile(columns_path(flow_dir), "r"))

def load_metadata(flow_dir):
    return metadata_io.read_metadata(transformed_metadata_dir(flow_dir))

def load_transform_output(flow_dir):
    return tft.TFTransformOutput(transform_artefacts_dir(flow_dir))

def train_data_files_path(flow_dir):
    return os.path.join(transformed_data_dir(flow_dir), "train-*.tfrecords")

def eval_data_files_path(flow_dir):
    return os.path.join(transformed_data_dir(flow_dir), "eval-*.tfrecords")

def cloud_train_job_launcher_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','bash_launchers',"train_on_cloud.sh")

def cloud_training_static_job_name(flow_dir, model_name):
    return f"train_{flow_dir}_{model_name}"

def cloud_training_job_name(flow_dir, model_name):
    return cloud_training_static_job_name(flow_dir, model_name)+"_"+current_time()

def hypertune_config_path(flow_dir):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','configs',"runs", 'hypertune', f"{flow_dir}.yaml")
    return path if os.path.isfile(path) else None

def best_model_dir_path(flow_dir, model_name):
    return json.load(tf.io.gfile.GFile(best_model_metadata_path(flow_dir, model_name), "r"))["best_model_path"]
        
def saved_model_dir(flow_dir, model_name):
    export_path = os.path.join(best_model_dir_path(flow_dir, model_name), 'export')
    return os.path.join(export_path, tf.io.gfile.listdir(export_path)[0])

def bigquery_model_results_table(flow_dir, model_name, schema=True):
    if schema:
        return f"{config.PROJECT}:{config.BQ_MODEL_RESULTS_DS}.{flow_dir}_{model_name}"
    else:
        return f"{config.PROJECT}.{config.BQ_MODEL_RESULTS_DS}.{flow_dir}_{model_name}"
    
def best_model_metadata_path(flow_dir, model_name):
    return os.path.join(model_runs_dir(flow_dir, model_name), "best", "metadata.json")

def ai_platform_job_metadata_path(flow_dir, model_name):
    return os.path.join(model_runs_dir(flow_dir, model_name), "ai_platform_job_metadata.json")