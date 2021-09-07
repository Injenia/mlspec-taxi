from configs.base_run import Run
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

class TaxiTS2017PosCompanyV01Run(Run):
    flow_dir   = "TaxiTS2017PosCompanyV01"
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
        company,

        # labels
        trip_seconds,
        #trip_total,

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
    label_column        = "trip_seconds"
    split_column        = "split_set"
    numeric_columns     = ['pickup_latitude',
                           'pickup_longitude',
                           'dropoff_latitude',
                           'dropoff_longitude']
    categorical_columns = ['company']
    
    PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS = 20
    
    def preprocess_fn(self, input_features):
        NUM_BUCKETS = self.PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS
        
        output_features = {}
        output_features[self.label_column] = input_features[self.label_column]
        
        for c in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']:
            output_features[f'{c}_bucketized'] = tft.bucketize(input_features[c], NUM_BUCKETS)
            
        #distance preprocessing
        l1_distance =   tf.abs(input_features['pickup_latitude']-input_features['dropoff_latitude']
                      )+tf.abs(input_features['pickup_longitude']-input_features['dropoff_longitude'])
        output_features['l1_distance'] = tft.scale_to_z_score(l1_distance)
        
        output_features['company_index'] = tft.compute_and_apply_vocabulary(input_features['company'], vocab_filename='company')
        
        return output_features
    
    
    def get_default_input_hparams(self):
        return dict(
            pickup_bucket_size             = self.PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS**2, # BUCKETS=20 => 400
            pickup_embed_dimension         = 2**int(np.log2(self.PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS**2)/2), # BUCKETS=20 => 16
            dropoff_bucket_size            = self.PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS**2, # BUCKETS=20 => 400
            dropoff_embed_dimension        = 2**int(np.log2(self.PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS**2)/2), # BUCKETS=20 => 16
            pickup_dropoff_bucket_size     = self.PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS**4, # BUCKETS=20 => 160000
            pickup_dropoff_embed_dimension = 2**int(np.log2(self.PICKUP_DROPOFF_LAT_LON_NUM_BUCKETS**4)/2), # BUCKETS=20 => 256
            company_hash_bucket_size       = 32,
            company_embed_dimension        = 8
        )
    
    
    def extended_feature_columns_fn(self, input_hparams):
        extended_wide_feature_columns=[]
        extended_deep_feature_columns=[]
        
        pickup_latitude_X_longitude = tf.feature_column.crossed_column(
            ['pickup_latitude_bucketized', 'pickup_longitude_bucketized'], 
            input_hparams.pickup_bucket_size)
        pickup_latitude_X_longitude_embedded = tf.feature_column.embedding_column(
            pickup_latitude_X_longitude, input_hparams.pickup_embed_dimension)
        extended_wide_feature_columns.append(pickup_latitude_X_longitude)
        extended_deep_feature_columns.append(pickup_latitude_X_longitude_embedded)
        
        
        dropoff_latitude_X_longitude = tf.feature_column.crossed_column(
            ['dropoff_latitude_bucketized', 'dropoff_longitude_bucketized'], 
            input_hparams.dropoff_bucket_size)
        dropoff_latitude_X_longitude_embedded = tf.feature_column.embedding_column(
            dropoff_latitude_X_longitude, input_hparams.dropoff_embed_dimension)
        extended_wide_feature_columns.append(dropoff_latitude_X_longitude)
        extended_deep_feature_columns.append(dropoff_latitude_X_longitude_embedded)
        
        
        pickup_X_dropoff = tf.feature_column.crossed_column(
            [pickup_latitude_X_longitude, dropoff_latitude_X_longitude], 
            input_hparams.pickup_dropoff_bucket_size)
        pickup_X_dropoff_embedded = tf.feature_column.embedding_column(
            pickup_X_dropoff, input_hparams.pickup_dropoff_embed_dimension)
        extended_wide_feature_columns.append(pickup_X_dropoff)
        extended_deep_feature_columns.append(pickup_X_dropoff_embedded)
        
        
        #company_embedded = tf.feature_column.embedding_column(
        #    'company_index', input_hparams.company_embed_dimension)
        #extended_deep_feature_columns.append(company_embedded)
        # NOTE: using this tf.feature_column.embedding_column causes the following error:
        # ValueError: Received a feature column from TensorFlow v1, but this is a TensorFlow v2 Estimator. Please either use v2 feature columns (accessible via tf.feature_column.* in TF 2.x) with this Estimator, or switch to a v1 Estimator for use with v1 feature columns (accessible via tf.compat.v1.estimator.* and tf.compat.v1.feature_column.*, respectively
        
        
        
        return extended_wide_feature_columns, extended_deep_feature_columns
    
    
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