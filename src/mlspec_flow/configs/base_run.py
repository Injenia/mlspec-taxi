from configs.globals import config
from utils.generic import dynamic_import
from utils import paths
from abc import ABC, abstractmethod
import argparse

class Run(ABC):
    flow_dir = ""
    model_name = ""
    query = ""
    key_column = ""
    label_column = ""
    split_column = ""
    numeric_columns = []
    categorical_columns = []
    
    def prepare_fn(self, bq_row):
        return bq_row
    
    @abstractmethod
    def preprocess_fn(self, input_features):
        pass
    
    def extended_feature_columns_fn(self, input_hparams):
        extended_wide_feature_columns=[]
        extended_deep_feature_columns=[]
        return extended_wide_feature_columns, extended_deep_feature_columns
    
    def initialize(self):
        from initializer.init import InitStep
        InitStep(
            self.flow_dir, 
            self.query, 
            self.key_column, 
            self.label_column, 
            self.split_column, 
            self.numeric_columns, 
            self.categorical_columns)
        
    def preprocess(self, debug = False, overwrite_previous = False):
        from preprocessor.preprocess import PreprocessStep
        PreprocessStep(
            self.flow_dir
        ).preprocess_data(
            self.preprocess_fn, 
            self.prepare_fn, 
            debug = debug,
            force = overwrite_previous)
       
    
    def get_default_input_hparams(self):
        return dict()
        
    def get_default_model_hparams(self):
        return dict()
        
    def _get_input_hparams(self, cli_params):
        parser   = argparse.ArgumentParser()
        defaults = self.get_default_input_hparams()
        for k,v in defaults.items():
            parser.add_argument(f"--{k}", type = type(v), default = v)
        hparams, unknown = parser.parse_known_args(cli_params)
        return hparams, unknown
        
        
    def train(self, cli_params, overwrite_previous = False):
        from trainer.train import TrainWideAndDeepEstimatorStep
        input_hparams, model_hparams = self._get_input_hparams(cli_params)
        default_model_hparams        = self.get_default_model_hparams()
        TrainWideAndDeepEstimatorStep(
            self.flow_dir, 
            self.model_name,
            input_hparams   = input_hparams,
            cli_params      = model_hparams,
            **default_model_hparams
        ).train_model(
            extended_feature_columns_fn = self.extended_feature_columns_fn, 
            force                       = overwrite_previous
        )
        
        
    def _run_name(self):
        name = type(self).__name__
        return name[:-3] if name.endswith("Run") else name
        
        
    def train_on_cloud(self, cli_params, overwrite_previous = False):
        # launching the training this way makes it easier to correctly pass 
        # the required parameters while delegating the package build to the 
        # "gcloud ai-platform jobs submit training" command in train_on_cloud.sh
        import subprocess
        import logging
        bash_file    = paths.cloud_train_job_launcher_path()
        run_name     = self._run_name()
        job_name     = paths.cloud_training_job_name(self.flow_dir, self.model_name)
        job_dir      = paths.model_runs_dir(self.flow_dir, self.model_name)
        hp_file      = paths.hypertune_config_path(self._run_name())
        region       = config.REGION
        force        = [ '--force' if overwrite_previous else '']
        run          = [f'--run={run_name}']
        hp           = "" if hp_file is None else f'--config={hp_file}'
        command      = ['bash', bash_file, job_name, job_dir, region, hp] + run + cli_params + force
        logging.info("launching")
        logging.info("\n".join(command))
        result       = subprocess.run(command, capture_output=True, text=True)
        logging.info(result.stdout)
        if len(result.stderr) > 0:
            logging.error(result.stderr)
            
            
    def select_best_model(self, overwrite_previous = False):
        from selector.select_best_model import SelectBestModelStep
        SelectBestModelStep(
            self.flow_dir, 
            self.model_name
        ).save_best_model_metadata(overwrite_previous = overwrite_previous)
        
        
    def validate(self, overwrite_previous = False):
        from validator.validate import ValidateStep
        ValidateStep(
            flow_dir   = self.flow_dir, 
            model_name = self.model_name, 
            batch_size = config.DF_BATCH_SIZE
        ).predict_data(
            self.prepare_fn, 
            overwrite_previous = overwrite_previous
        )
        
        
    def evaluate(self):
        from evaluator.evaluate import BareMinimumEvaluatorStep
        BareMinimumEvaluatorStep(
            flow_dir   = self.flow_dir, 
            model_name = self.model_name,
        ).evaluate()
        
    
        
def get_run(run_name):
    run_class = dynamic_import(f"configs.runs.{run_name}",f"{run_name}Run")
    run_instance = run_class()
    return run_instance