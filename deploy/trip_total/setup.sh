#!/bin/bash

set -e
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${current_dir}/config.sh

gcloud ai-platform models create ${AI_PLATFORM_PREDICTION_MODEL} --region=${AI_PLATFORM_PREDICTION_REGION}  --enable-logging
gcloud ai-platform versions create ${AI_PLATFORM_PREDICTION_VERSION} \
--region=${AI_PLATFORM_PREDICTION_REGION} \
--model=${AI_PLATFORM_PREDICTION_MODEL} \
--origin=${MODELDIR} \
--runtime-version=2.4