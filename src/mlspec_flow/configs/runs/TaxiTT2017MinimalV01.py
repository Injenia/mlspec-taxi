from configs.base_run import Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTT2017MinimalV01Run(Run):
    flow_dir   = "TaxiTT2017MinimalV01"
    model_name = "20210521"
    query      = """SELECT 
        # IDs, these will not be features, we keep them for record traceability
        unique_key,
        #taxi_id,

        # raw features (cast to string is for cathegorical data)
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude,

        # labels
        #trip_seconds,
        trip_total,

        # data split
        case EXTRACT(YEAR FROM trip_start_timestamp)
            when 2018 then "VALIDATE"
            when 2019 then "TEST"
            when 2020 then "COVID19"
            when 2021 then "COVID19"
            else "TRAIN"
        END as split_set

        # fields not availbale at prediction time: trip_end_timestamp, fare, tips, tolls, extras, payment_type

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
        where
        # reproducibility constraints
            trip_start_timestamp <= TIMESTAMP("2021-03-01 00:00:00 UTC")
        AND trip_start_timestamp >= TIMESTAMP("2013-01-01 00:00:00 UTC")

        # label constraints
        AND trip_seconds is not null
        AND trip_seconds > 0
        AND trip_total   is not null
        AND trip_total   > 0

        # feature nullability constraints
        AND trip_miles             is not null
        AND pickup_census_tract    is not null
        AND dropoff_census_tract   is not null
        AND pickup_community_area  is not null
        AND dropoff_community_area is not null
        AND company                is not null
        AND pickup_latitude        is not null
        AND pickup_longitude       is not null
        AND dropoff_latitude       is not null
        AND dropoff_longitude      is not null
        AND EXTRACT(YEAR from trip_start_timestamp) >= 2017 # train set limited to year 2017
        and trip_miles   > 0 
        AND trip_seconds > 0 
        AND trip_total   > 0  
        AND ABS(dropoff_latitude-pickup_latitude) + ABS(dropoff_longitude-pickup_longitude) > 0 # L1 distance
    """
    key_column          = "unique_key"
    label_column        = "trip_total"
    split_column        = "split_set"
    numeric_columns     = ['pickup_latitude',
                           'pickup_longitude',
                           'dropoff_latitude',
                           'dropoff_longitude']
    categorical_columns = []
    
    def preprocess_fn(self, input_features):
        output_features = {}
        output_features[self.label_column] = input_features[self.label_column]
        
        #distance preprocessing
        l1_distance =   tf.abs(input_features['pickup_latitude']-input_features['dropoff_latitude']
                      )+tf.abs(input_features['pickup_longitude']-input_features['dropoff_longitude'])
        output_features['l1_distance'] = tft.scale_to_z_score(l1_distance)
        return output_features
    
    
    def get_default_model_hparams(self):
        batch_size           = 1000000
        epoch_size           = 13946748
        learning_rate        = 0.01
        num_epochs           = 15
        num_evaluations      = 15
        epoch_steps          = np.ceil(batch_size/epoch_size).astype(int)
        evaluate_every_epoch = int(num_epochs/num_evaluations)
        return dict(
            num_epochs           = num_epochs,
            batch_size           = batch_size, 
            learning_rate        = learning_rate,
            evaluate_every_epoch = evaluate_every_epoch
        )