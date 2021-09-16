# Chicago Taxi Trips

## Dataset download

Dateset is hosted in BQ: `bigquery-public-data:chicago_taxi_trips`  
Kaggle homepage: [here](https://www.kaggle.com/chicago/chicago-taxi-trips-bq)  
Dataset homepage: [here](https://digital.cityofchicago.org/index.php/chicago-taxi-data-released/)



## EDA

**001**: analysis of null values, Distribution of both targets, Correlations between columns  
**002**: Correlation of engineered features (L1,L2 distances) with other columns  
**003**: pickup/dropoff data ranges  
**004, 005**: Company field analysis  

## Training

### **Setup**

```bash
pip install -r requirements.txt
cd src/mlspec_flow
```

### **Preparation**

Choose a run from the ones available in `src/mlspec_flow/configs/runs/`  
Runs that are ultimately reported in the whitepaper are:

- TaxiTS2017MinimalV01 (TripSeconds)
- TaxiTT2017PosTimeCompanyV01 (TripTotal)

```bash
export RUN=TaxiTT2017PosTimeCompanyV01 
```

### **Full training process is made of 5 steps**

__Initialization__: Performs GCS setup

```bash
python run.py initialize $RUN
```

__Preprocessing__: Launches a dataflow pipeline that generates TFRrecords

```bash
python run.py preprocess $RUN
```

__Training__: Launches an AI Platform job that trains the model

```bash
python run.py train $RUN
```

__Best model selection__: Selects the best model (according to the evaluation metric) produced by the HPTuning, if no HPTuning is performed, the selected model is just the result of the training

```bash
python run.py select_best_model $RUN
```

__Validation__: Launches a Dataflow pipeline that runs the exported model on the test set

```bash
python run.py validate $RUN
```

__Evaluation__: Prints model performances calculated on the validation results 

```bash
python run.py evaluate $RUN
```

## Model deployment

### **Preparation**

Access the correct folder under *deploy* depending on the task

```bash
# for trip seconds
cd deploy/trip_seconds

# for trip_total
cd deploy/trip_total
```

Modify **config.sh**

```bash
#!/bin/bash

MODELDIR= # the exported model directory (in standard tensorflow saved_model format)
AI_PLATFORM_PREDICTION_REGION= # AI Platform region of deployment
AI_PLATFORM_PREDICTION_MODEL= # name of the AI Platform Predicions Model
AI_PLATFORM_PREDICTION_VERSION= # name of the AI Platform Predicions Model's Version
```

Note that $MODELDIR values default to the two models reported in the whitepaper for the two tasks respectively.

### Deployment

Now to deploy the model, simply invoke

```bash
bash setup.sh
```

This will create AI Platform Predictions model and deploy the neural network to the version as specified in **config.sh**

Once the script has completed, the deployed model can be tested with:

```bash
bash invoke.sh
```

Which will perform a REST API towards the newly deployed model.

### Cleanup

To remove the deployed model from the cloud, execute

```bash
bash teardown.sh
```

This will remove both version and model from Ai Platform Predictions.
