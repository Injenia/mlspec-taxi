#!/bin/bash

current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
now=$(date +"%Y%m%d%H%M%S")

#cd ${current_dir}'/mlspec_flow'

gcloud ai-platform jobs submit training 'train'${now} \
        --scale-tier basic_gpu \
        --package-path ${current_dir}'/mlspec_flow/trainer/' \
        --module-name 'trainer.task' \
        --job-dir 'gs://mlteam-ml-specialization-2021-taxi/mlengine_jobs/'${now} \
        --region 'europe-west1' \
        --runtime-version '2.4' \
        --python-version '3.7' \
        -- \
        $@