#!/bin/bash

job_name=$1
shift
job_dir=$1
shift
region=$1
shift
config=$1
shift
args=$@


current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
launch_dir=${current_dir}'/..'
package_path=${launch_dir}'/mlspec_flow/trainer/'

cd ${launch_dir}

gcloud ai-platform jobs submit training ${job_name} \
        --scale-tier basic_gpu \
        --package-path ${package_path} \
        --module-name 'trainer.task' \
        --job-dir ${job_dir} \
        --region ${region} \
        --runtime-version '2.4' \
        --python-version '3.7' \
        ${config} \
        -- \
        ${args}