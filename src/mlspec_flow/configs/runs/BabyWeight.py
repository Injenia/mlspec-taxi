from configs.base_run import Run
import tensorflow as tf
import tensorflow_transform as tft

class BabyWeightRun(Run):
    flow_dir = "BabyWeight"
    model_name = "20210409V01"
    query = """SELECT
    ROUND(weight_pounds,1) AS weight_pounds,
    is_male,
    mother_age,
    mother_race,
    plurality,
    gestation_weeks,
    FARM_FINGERPRINT( 
        CONCAT(
          COALESCE(CAST(weight_pounds AS STRING), 'NA'),
          COALESCE(CAST(is_male AS STRING),'NA'),
          COALESCE(CAST(mother_age AS STRING),'NA'),
          COALESCE(CAST(mother_race AS STRING),'NA'),
          COALESCE(CAST(plurality AS STRING), 'NA'),
          COALESCE(CAST(gestation_weeks AS STRING),'NA')
          )
        ) AS key,
    IF(MOD(
        FARM_FINGERPRINT( 
            CONCAT(
                COALESCE(CAST(weight_pounds AS STRING), 'NA'),
                COALESCE(CAST(is_male AS STRING),'NA'),
                COALESCE(CAST(mother_age AS STRING),'NA'),
                COALESCE(CAST(mother_race AS STRING),'NA'),
                COALESCE(CAST(plurality AS STRING), 'NA'),
                COALESCE(CAST(gestation_weeks AS STRING),'NA')
                )
    ), 100) < 70, "TRAIN", "VALIDATE") as split_column,
    FROM
        publicdata.samples.natality
    WHERE year > 1000
    AND weight_pounds > 0
    AND mother_age > 0
    AND plurality > 0
    AND gestation_weeks > 0
    AND month > 0
    limit 2000
    """
    key_column = "key"
    label_column = "weight_pounds"
    split_column = "split_column"
    numeric_columns = ['mother_age', 'plurality', 'gestation_weeks']
    categorical_columns = ['is_male', 'mother_race']
    
    
    def prepare_fn(self, bq_row):
        # modify opaque numeric race code into human-readable data
        races = dict(zip([1,2,3,4,5,6,7,18,28,39,48],
                         ['White', 'Black', 'American Indian', 'Chinese', 
                          'Japanese', 'Hawaiian', 'Filipino',
                          'Asian Indian', 'Korean', 'Samaon', 'Vietnamese']))
        result = {} 
        for feature_name in bq_row.keys():
            result[feature_name] = str(bq_row[feature_name]) if feature_name in self.categorical_columns else bq_row[feature_name]
        if 'mother_race' in bq_row and bq_row['mother_race'] in races:
            result['mother_race'] = races[bq_row['mother_race']]
        else:
            result['mother_race'] = 'Unknown'

        return result
    
    
    def preprocess_fn(self, input_features):
        output_features = {}
        # target feature
        output_features['weight_pounds'] = input_features['weight_pounds']
        # normalisation
        output_features['mother_age_normalized'] = tft.scale_to_z_score(input_features['mother_age'])
        output_features['gestation_weeks_normalized'] =  tft.scale_to_0_1(input_features['gestation_weeks'])
        # bucktisation based on quantiles
        output_features['mother_age_bucketized'] = tft.bucketize(input_features['mother_age'], num_buckets=5)
        # you can compute new features based on custom formulas
        output_features['mother_age_log'] = tf.math.log(input_features['mother_age'])
        # or create flags/indicators
        is_multiple = tf.as_string(input_features['plurality'] > tf.constant(1.0))
        # convert categorical features to indexed vocab
        output_features['mother_race_index'] = tft.compute_and_apply_vocabulary(input_features['mother_race'], vocab_filename='mother_race')
        output_features['is_male_index'] = tft.compute_and_apply_vocabulary(input_features['is_male'], vocab_filename='is_male')
        output_features['is_multiple_index'] = tft.compute_and_apply_vocabulary(is_multiple, vocab_filename='is_multiple')
        return output_features
    
    def get_default_input_hparams(self):
        return dict(embed_dimension=5)
    
    def extended_feature_columns_fn(self, input_hparams):
        extended_wide_feature_columns=[]
        extended_deep_feature_columns=[]
        mother_race_X_mother_age_bucketized = tf.feature_column.crossed_column(
            ['mother_age_bucketized', 'mother_race_index'],  55)
        extended_wide_feature_columns.append(mother_race_X_mother_age_bucketized)
        mother_race_X_mother_age_bucketized_embedded = tf.feature_column.embedding_column(
            mother_race_X_mother_age_bucketized, input_hparams.embed_dimension)#["embed_dimension"])
        extended_deep_feature_columns.append(mother_race_X_mother_age_bucketized_embedded)
        return extended_wide_feature_columns, extended_deep_feature_columns
    
    def get_default_model_hparams(self):
        return dict(
            num_epochs = 10
        )