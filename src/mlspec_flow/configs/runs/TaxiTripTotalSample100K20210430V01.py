from configs.base_run import Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

constraints="""
        
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
"""

class TaxiTripTotalSample100K20210430V01Run(Run):
    flow_dir   = "TaxiTripTotalSample100K20210430V01"
    model_name = "20210430V01"
    query      = f"""SELECT 
        # IDs, these will not be features, we keep them for record traceability
        unique_key,

        # raw features (cast to string is for cathegorical data)
        cast(pickup_census_tract    as string) as pickup_census_tract,
        cast(dropoff_census_tract   as string) as dropoff_census_tract,
        cast(pickup_community_area  as string) as pickup_community_area,
        cast(dropoff_community_area as string) as dropoff_community_area,
        FORMAT_TIMESTAMP("%Y-%m-%d %a-%H:%M", trip_start_timestamp, "UTC") as trip_start,
        company,
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

        FROM (
            (select * from `bigquery-public-data.chicago_taxi_trips.taxi_trips` where EXTRACT(YEAR FROM trip_start_timestamp) < 2018 and {constraints} limit 100000) UNION ALL
            (select * from `bigquery-public-data.chicago_taxi_trips.taxi_trips` where EXTRACT(YEAR FROM trip_start_timestamp) = 2018 and {constraints} limit 10000)  UNION ALL
            (select * from `bigquery-public-data.chicago_taxi_trips.taxi_trips` where EXTRACT(YEAR FROM trip_start_timestamp) = 2019 and {constraints} limit 10000)  UNION ALL
            (select * from `bigquery-public-data.chicago_taxi_trips.taxi_trips` where EXTRACT(YEAR FROM trip_start_timestamp) > 2019 and {constraints} limit 10000) 
        )
    """
    key_column          = "unique_key"
    label_column        = "trip_total"
    split_column        = "split_set"
    numeric_columns     = ['trip_miles', 
                           'pickup_latitude',
                           'pickup_longitude',
                           'dropoff_latitude',
                           'dropoff_longitude']
    categorical_columns = ['company',
                           'pickup_census_tract',
                           'dropoff_census_tract',
                           'pickup_community_area',
                           'dropoff_community_area']
    key_column          = "unique_key"
    label_column        = "trip_total"
    split_column        = "split_set"
    numeric_columns     = ['pickup_latitude',
                           'pickup_longitude',
                           'dropoff_latitude',
                           'dropoff_longitude']
    categorical_columns = ['company',
                           'pickup_census_tract',
                           'dropoff_census_tract',
                           'pickup_community_area',
                           'dropoff_community_area',
                           'trip_start' # [technical] this is in categorical because its placeholder for serving must be of type string
                          ]
    
    
    def preprocess_fn(self, input_features):
        output_features = {}
        output_features[self.label_column] = input_features[self.label_column]
        for c in self.numeric_columns:
            output_features[c] = tft.scale_to_z_score(input_features[c])
        for c in self.categorical_columns:
            if c not in ['trip_start']:
                output_features[f'{c}_index'] = tft.compute_and_apply_vocabulary(input_features[c], vocab_filename=c)
                
        # date preprocessing
        def tf_strings_split(s, sep, len):
            return [tf.strings.split(s, sep=sep).to_tensor()[:,l] for l in range(len)]
        
        date_and_time             = input_features['trip_start']
        date, day_and_time        = tf_strings_split(date_and_time, sep=" ", len=2)
        year, month, day_of_month = tf_strings_split(date,          sep="-", len=3)
        day_of_week, time         = tf_strings_split(day_and_time,  sep="-", len=2)
        hour, minutes             = tf_strings_split(time,          sep=":", len=2)
        
        output_features[f'month_index']        = tft.compute_and_apply_vocabulary(month,        vocab_filename='month')
        output_features[f'day_of_month_index'] = tft.compute_and_apply_vocabulary(day_of_month, vocab_filename='day_of_month')
        output_features[f'day_of_week_index']  = tft.compute_and_apply_vocabulary(day_of_week,  vocab_filename='day_of_week')
        output_features[f'hour_index']         = tft.compute_and_apply_vocabulary(hour,         vocab_filename='hour')
        
        #distance preprocessing
        l1_distance =   tf.abs(input_features['pickup_latitude']-input_features['dropoff_latitude']
                      )+tf.abs(input_features['pickup_longitude']-input_features['dropoff_longitude'])
        output_features['l1_distance'] = tft.scale_to_z_score(l1_distance)
        
        return output_features
    
    def get_default_input_hparams(self):
        return dict(
            census_tract_hash_bucket_size   = 128,
            census_tract_embed_dimension    = 16,
            community_area_hash_bucket_size = 128,
            community_area_embed_dimension  = 16,
            day_and_hour_hash_bucket_size   = 128,
            day_and_hour_embed_dimension    = 16,
            
        )
    
    def extended_feature_columns_fn(self, input_hparams):
        extended_wide_feature_columns=[]
        extended_deep_feature_columns=[]
        
        census_tract_pickup_X_dropoff_bucketized = tf.feature_column.crossed_column(
            ['pickup_census_tract_index', 'dropoff_census_tract_index'], input_hparams.census_tract_hash_bucket_size)
        census_tract_pickup_X_dropoff_bucketized_embedded = tf.feature_column.embedding_column(
            census_tract_pickup_X_dropoff_bucketized, input_hparams.census_tract_embed_dimension)
        extended_wide_feature_columns.append(census_tract_pickup_X_dropoff_bucketized)
        extended_deep_feature_columns.append(census_tract_pickup_X_dropoff_bucketized_embedded)
        
        community_area_pickup_X_dropoff_bucketized = tf.feature_column.crossed_column(
            ['pickup_community_area_index', 'dropoff_community_area_index'], input_hparams.community_area_hash_bucket_size)
        community_area_pickup_X_dropoff_bucketized_embedded = tf.feature_column.embedding_column(
            community_area_pickup_X_dropoff_bucketized, input_hparams.community_area_embed_dimension)
        extended_wide_feature_columns.append(community_area_pickup_X_dropoff_bucketized)
        extended_deep_feature_columns.append(community_area_pickup_X_dropoff_bucketized_embedded)
        
        day_of_week_X_hour_bucketized = tf.feature_column.crossed_column(
            ['day_of_week_index', 'hour_index'], input_hparams.day_and_hour_hash_bucket_size)
        day_of_week_X_hour_bucketized_embedded = tf.feature_column.embedding_column(
            day_of_week_X_hour_bucketized, input_hparams.day_and_hour_embed_dimension)
        extended_wide_feature_columns.append(day_of_week_X_hour_bucketized)
        extended_deep_feature_columns.append(day_of_week_X_hour_bucketized_embedded)
        
        return extended_wide_feature_columns, extended_deep_feature_columns
    
    
    def get_default_model_hparams(self):
        batch_size    = 100000
        epoch_size    = 100000
        learning_rate = 0.01
        epoch_steps   = np.ceil(batch_size/epoch_size).astype(int)
        return dict(
            evaluate_after_sec = 1,
            num_epochs         = 100,
            batch_size         = batch_size, 
            learning_rate      = learning_rate,
            decay_steps        = epoch_steps, # decay every epoch
            evaluate_every_epoch = 10
        )