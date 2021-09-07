class Config:
    PROJECT              = 'mlteam-ml-specialization-2021'
    BUCKET               = 'mlteam-ml-specialization-2021-taxi'
    REGION               = 'europe-west1'
    ROOT_DIR             = 'tft_flows'
    JOB_NAME             = 'data-tft'
    DATAFLOW_STAGING_DIR = 'gs://mlteam-ml-specialization-2021-taxi/tmp/staging'
    DATAFLOW_TEMP_DIR    = 'gs://mlteam-ml-specialization-2021-taxi/tmp/temp'
    BQ_MODEL_RESULTS_DS  = 'models_results'
    DF_BATCH_SIZE        = 500
    
    
config = Config()