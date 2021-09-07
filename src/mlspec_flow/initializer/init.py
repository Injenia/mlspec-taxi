import tensorflow as tf
import json

from utils.paths import *
from configs.globals import config

def InitStep(flow_dir, query, key_column, label_column, split_column, numeric_columns, categorical_columns):
    assert not tf.io.gfile.exists(flow_dir), f"{flow_dir} already exists"
    tf.io.gfile.GFile(query_path(flow_dir), "w").write(query)
    columns = dict(
        key_column=key_column,
        label_column=label_column, 
        split_column=split_column,
        numeric_columns=numeric_columns, 
        categorical_columns=categorical_columns
    )
    json.dump(columns, tf.io.gfile.GFile(columns_path(flow_dir), "w"))