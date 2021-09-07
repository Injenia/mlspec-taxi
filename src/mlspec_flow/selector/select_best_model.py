import tensorflow as tf
import json

from utils.paths import *
from configs.globals import config


class SelectBestModelStep:
    def __init__(self, flow_dir, model_name):
        self.flow_dir = flow_dir
        self.model_name = model_name
        
        
    def _ai_platform_job_data(self):
        metadata_path = ai_platform_job_metadata_path(self.flow_dir, self.model_name)
        if tf.io.gfile.exists(metadata_path):
            return json.load(tf.io.gfile.GFile(metadata_path, "r"))
        
        from oauth2client.client import GoogleCredentials
        from googleapiclient import discovery
        from googleapiclient import errors

        ml = discovery.build('ml','v1')
        projectId = 'projects/{}'.format(config.PROJECT)
        ai_platform_data = ml.projects().jobs().list(parent=projectId).execute()
        json.dump(ai_platform_data, tf.io.gfile.GFile(metadata_path, "w"))
        return ai_platform_data
        
        
        
    def _hypertuning_data(self):
        flow_dir, model_name = self.flow_dir, self.model_name
        
        from oauth2client.client import GoogleCredentials
        from googleapiclient import discovery
        from googleapiclient import errors

        ai_platform_data = self._ai_platform_job_data()
        jobs = ai_platform_data["jobs"]
        succeeded_jobs = [j for j in jobs if j["state"] == "SUCCEEDED"]
        job_name = cloud_training_static_job_name(self.flow_dir, self.model_name)
        latest_job = sorted([j for j in jobs if j['jobId'].startswith(job_name)], key=lambda x: x["jobId"])[-1]
        if latest_job["trainingOutput"].get('isHyperparameterTuningJob',None) is not None:
            trials = sorted(latest_job["trainingOutput"]["trials"], key=lambda x: x["finalMetric"]["objectiveValue"])
            trial = trials[0 if latest_job["trainingInput"]["hyperparameters"]["goal"]=="MINIMIZE" else -1]
            trial_id = trial["trialId"]
        else:
            trial_id = "1"
        return ai_platform_data, os.path.join(model_runs_dir(self.flow_dir, self.model_name), trial_id)
        
    def save_best_model_metadata(self, overwrite_previous = False):
        metadata_path   = best_model_metadata_path(self.flow_dir, self.model_name)
        if not overwrite_previous:
            assert not tf.io.gfile.exists(metadata_path), f"{metadata_path} already exists"
        ai_platform_data, best_model_path = self._hypertuning_data()
        metadata = dict(best_model_path=best_model_path)
        json.dump(metadata, tf.io.gfile.GFile(metadata_path, "w"))
        