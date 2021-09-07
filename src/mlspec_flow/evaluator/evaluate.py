from utils.paths import *
import pandas as pd
import logging

def default_result_handler(result):
    from pprint import pprint
    pprint(pd.DataFrame(result).set_index('split_set').transpose())

class BareMinimumEvaluatorStep:
    def __init__(self, flow_dir, model_name):
        self.flow_dir   = flow_dir
        self.model_name = model_name
        self.table      = bigquery_model_results_table(flow_dir, model_name, schema=False)
        self.columns    = load_columns(flow_dir)
        
    def evaluate(self, result_handler = default_result_handler):
        query = f"""
        SELECT 
        split_set, 
        avg(abs(prediction - {self.columns["label_column"]})) as mae, 
        avg(abs(baseline   - {self.columns["label_column"]})) as baseline_mae,
        avg(abs(prediction - {self.columns["label_column"]})/{self.columns["label_column"]}) as mre, 
        avg(abs(baseline   - {self.columns["label_column"]})/{self.columns["label_column"]}) as baseline_mre,
        FROM `{self.table}` 
        group by 1
        """
        result = pd.read_gbq(query).to_dict(orient='records')
        default_result_handler(result)