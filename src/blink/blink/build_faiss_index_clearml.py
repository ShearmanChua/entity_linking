from clearml import Task,TaskTypes,Dataset

PROJECT_TASK_NAME = "BLINK/task"
PROJECT_DATASET_NAME = "BLINK/dataset"
TASK_NAME = "generate FAISS index"
INPUT_DATASET = "BLINK_models"
OUTPUT_DATASET = "FAISS_index"

task = Task.init(project_name=PROJECT_TASK_NAME, task_name=TASK_NAME)
task.set_base_docker("nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04",docker_setup_bash_script=['pip install -e git+https://github.com/facebookresearch/BLINK.git#egg=BLINK'])

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import logging
import numpy
import os
import time
import torch

from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import blink.candidate_ranking.utils as utils

logger = utils.get_logger()

def main(params): 
    output_path = params["output_path"]
    output_dir, _ = os.path.split(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = utils.get_logger(output_dir)

    logger.info("Loading candidate encoding from path: %s" % params["candidate_encoding"])
    datasets_dict = Dataset.list_datasets(
        dataset_project=PROJECT_DATASET_NAME,
        partial_name=INPUT_DATASET,
        only_completed=False,
    )

    # Find the dataset that matches exact dataset name
    datasets_obj = [
        Dataset.get(dataset_dict["id"])
        for dataset_dict in datasets_dict
        if dataset_dict["name"] == INPUT_DATASET
    ]

    # Reverse list due to child-parent dependency, and get the first dataset object
    dataset_obj = datasets_obj[::-1][0]
    dataset_obj_prefix = dataset_obj.get_local_copy()

    candidate_encoding = torch.load(os.path.join(dataset_obj_prefix,params["candidate_encoding"]))
    vector_size = candidate_encoding.size(1)
    index_buffer = params["index_buffer"]
    if params["hnsw"]:
        logger.info("Using HNSW index in FAISS")
        index = DenseHNSWFlatIndexer(vector_size, index_buffer)
    else:
        logger.info("Using Flat index in FAISS")
        index = DenseFlatIndexer(vector_size, index_buffer)

    logger.info("Building index.")
    index.index_data(candidate_encoding.numpy())
    logger.info("Done indexing data.")

    if params.get("save_index", None):
        index.serialize(output_path)

def create_dataset(file_to_add, dataset_project, dataset_name):
    """
    Checks if parent dataset exists
    - if yes, finalise the parent dataset, then create new child dataset and point to parent
    - if no, create new dataset as parent
    """
    parent_dataset = _get_last_child_dataset(dataset_project, dataset_name)
    if parent_dataset:
        print("create child")
        try:
            parent_dataset.finalize()
        except:
            print("Cannot finalize parent dataset")
        child_dataset = Dataset.create(
            dataset_name, dataset_project, parent_datasets=[parent_dataset]
        )
        child_dataset.add_files(file_to_add)
        child_dataset.upload(output_url="s3://experiment-logging/storage")
        return child_dataset
    else:
        print("create parent")
        dataset = Dataset.create(dataset_name, dataset_project)
        child_dataset.add_files(file_to_add)
        dataset.upload(output_url="s3://experiment-logging/storage")
        return dataset


def _get_last_child_dataset(dataset_project, dataset_name):
    """Get last dataset child object"""
    datasets_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )
    if datasets_dict:
        datasets_dict_latest = datasets_dict[-1]
        return Dataset.get(dataset_id=datasets_dict_latest["id"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="output file path",
    )
    parser.add_argument(
        "--candidate_encoding",
        default="new_entities_large.t7",
        type=str,
        help="file path for candidte encoding.",
    )
    parser.add_argument(
        "--hnsw", action='store_true', 
        help='If enabled, use inference time efficient HNSW index',
    )
    parser.add_argument(
        "--save_index", action='store_true', 
        help='If enabled, save index',
    )
    parser.add_argument(
        '--index_buffer', type=int, default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )

    params = parser.parse_args()
    task.execute_remotely('compute')
    params = params.__dict__

    main(params)
