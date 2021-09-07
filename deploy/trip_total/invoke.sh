#!/bin/bash

current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source ${current_dir}/config.sh

curl -d '{"instances": [{"pickup_latitude": 0, "pickup_longitude": 0, "dropoff_latitude": 0, "dropoff_longitude": 0, "trip_start": "2017-03-03 Fry-09:00", "company": ""}], "signature_name": "predict"}' \
    -X POST https://${AI_PLATFORM_PREDICTION_REGION}-ml.googleapis.com/v1/projects/mlteam-ml-specialization-2021/models/${AI_PLATFORM_PREDICTION_MODEL}:predict?access_token\="$(gcloud auth application-default print-access-token)"