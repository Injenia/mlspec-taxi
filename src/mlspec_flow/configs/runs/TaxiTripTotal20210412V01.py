from configs.base_run import Run
import tensorflow as tf
import tensorflow_transform as tft

class TaxiTripTotal20210412V01Run(Run):
    flow_dir   = "TaxiTripTotal20210412V01"
    model_name = "20210412V01"
    query      = """SELECT 
        # IDs, these will not be features, we keep them for record traceability
        unique_key,
        #taxi_id,

        # raw features (cast to string is for cathegorical data)
        trip_miles,
        cast(pickup_census_tract    as string) as pickup_census_tract,
        cast(dropoff_census_tract   as string) as dropoff_census_tract,
        cast(pickup_community_area  as string) as pickup_community_area,
        cast(dropoff_community_area as string) as dropoff_community_area,
        company,
        pickup_latitude,
        pickup_longitude,
        --pickup_location,
        dropoff_latitude,
        dropoff_longitude,
        --dropoff_location,

        # derived features, --omitted for now
        --CAST(EXTRACT(MONTH FROM trip_start_timestamp) AS STRING) as month_start
        --CAST(EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS STRING) as dayofweek_start
        --ABS(pickup_latitude - dropoff_latitude) + ABS(pickup_longitude - dropoff_longitude) as manhattan_distance

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
        --AND pickup_location        is not null
        AND dropoff_latitude       is not null
        AND dropoff_longitude      is not null
        --AND dropoff_location       is not null
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
    
    # as a very first prototype, we implement the bare minimum required
    
    #def prepare_fn(self, bq_row):
    #    return bq_row
    
    
    def preprocess_fn(self, input_features):
        output_features = {}
        output_features[self.label_column] = input_features[self.label_column]
        for c in self.numeric_columns:
            output_features[c] = tft.scale_to_z_score(input_features[c])
        for c in self.categorical_columns:
            output_features[f'{c}_index'] = tft.compute_and_apply_vocabulary(input_features[c], vocab_filename=c)
        return output_features
    
    def get_default_input_hparams(self):
        return dict()
    
    def extended_feature_columns_fn(self, input_hparams):
        extended_wide_feature_columns=[]
        extended_deep_feature_columns=[]
        return extended_wide_feature_columns, extended_deep_feature_columns