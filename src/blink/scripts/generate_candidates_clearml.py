from clearml import Task,TaskTypes,Dataset

PROJECT_TASK_NAME = "BLINK/task"
PROJECT_DATASET_NAME = "BLINK/dataset"
TASK_NAME = "generate biencoder embeddings"
OUTPUT_DATASET = "BLINK_models"

Task.add_requirements("protobuf")
task = Task.init(project_name=PROJECT_TASK_NAME, task_name=TASK_NAME)
task.set_base_docker("nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04",docker_setup_bash_script=['pip install -e git+https://github.com/facebookresearch/BLINK.git#egg=BLINK'])
task.set_task_type(TaskTypes.inference)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from elq.biencoder.biencoder import load_biencoder
# import elq.candidate_ranking.utils as utils
from blink.biencoder.biencoder import load_biencoder
import blink.candidate_ranking.utils as utils
import json
import sys
import os
from tempfile import gettempdir
from tqdm import tqdm

import argparse


def encode_candidate(
    reranker,
    candidate_pool,
    encode_batch_size,
    silent,
    logger,
):
    reranker.model.eval()
    device = reranker.device
    #for cand_pool in candidate_pool:
    #logger.info("Encoding candidate pool %s" % src)
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


def load_candidate_pool(
    tokenizer,
    params,
    logger,
    cand_pool_path,
):
    candidate_pool = None
    # try to load candidate pool from file
    try:
        logger.info("Loading pre-generated candidate pool from: ")
        logger.info(cand_pool_path)
        candidate_pool = torch.load(cand_pool_path)
    except:
        logger.info("Loading failed.")
    assert candidate_pool is not None

    return candidate_pool

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


parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset_name', type=str, required=True, help='dataset containing models')
parser.add_argument('--path_to_model_config', type=str, required=True, help='filepath to saved model config')
parser.add_argument('--path_to_model', type=str, required=True, help='filepath to saved model')
parser.add_argument('--entity_dict_path', type=str, required=True, help='filepath to entities to encode (.jsonl file)')
parser.add_argument('--saved_cand_ids', type=str, help='filepath to entities pre-parsed into IDs')
parser.add_argument('--encoding_save_file_dir', type=str, help='directory of file to save generated encodings', default=None)
parser.add_argument('--test', action='store_true', default=False, help='whether to just test encoding subsample of entities')

parser.add_argument('--compare_saved_embeds', type=str, help='compare against these saved embeddings')

parser.add_argument('--batch_size', type=int, default=512, help='batch size for encoding candidate vectors (default 512)')

parser.add_argument('--chunk_start', type=int, default=0, help='example idx to start encoding at (for parallelizing encoding process)')
parser.add_argument('--chunk_end', type=int, default=-1, help='example idx to stop encoding at (for parallelizing encoding process)')


args = parser.parse_args()

task.execute_remotely('compute')

datasets_dict = Dataset.list_datasets(
    dataset_project=PROJECT_DATASET_NAME,
    partial_name=args.input_dataset_name,
    only_completed=False,
)

# Find the dataset that matches exact dataset name
datasets_obj = [
    Dataset.get(dataset_dict["id"])
    for dataset_dict in datasets_dict
    if dataset_dict["name"] == args.input_dataset_name
]

# Reverse list due to child-parent dependency, and get the first dataset object
dataset_obj = datasets_obj[::-1][0]
dataset_obj_prefix = dataset_obj.get_local_copy()

try:
    with open(os.path.join(dataset_obj_prefix,args.path_to_model_config)) as json_file:
        biencoder_params = json.load(json_file)
except json.decoder.JSONDecodeError:
    with open(os.path.join(dataset_obj_prefix,args.path_to_model_config)) as json_file:
        for line in json_file:
            line = line.replace("'", "\"")
            line = line.replace("True", "true")
            line = line.replace("False", "false")
            line = line.replace("None", "null")
            biencoder_params = json.loads(line)
            break
# model to use
biencoder_params["path_to_model"] = os.path.join(dataset_obj_prefix,args.path_to_model)
# entities to use
biencoder_params["entity_dict_path"] = os.path.join(dataset_obj_prefix,args.entity_dict_path)
biencoder_params["degug"] = False
biencoder_params["data_parallel"] = True
biencoder_params["no_cuda"] = False
biencoder_params["max_context_length"] = 32
biencoder_params["encode_batch_size"] = args.batch_size
# biencoder_params["encode_batch_size"] = 16
print(biencoder_params)

saved_cand_ids = os.path.join(dataset_obj_prefix,getattr(args, 'saved_cand_ids', None))
encoding_save_file_dir = gettempdir()
if encoding_save_file_dir is not None and not os.path.exists(encoding_save_file_dir):
    os.makedirs(encoding_save_file_dir, exist_ok=True)

logger = utils.get_logger(biencoder_params.get("model_output_path", None))
print(biencoder_params)
biencoder = load_biencoder(biencoder_params)
baseline_candidate_encoding = None
if getattr(args, 'compare_saved_embeds', None) is not None:
    baseline_candidate_encoding = torch.load(getattr(args, 'compare_saved_embeds'))

candidate_pool = load_candidate_pool(
    biencoder.tokenizer,
    biencoder_params,
    logger,
    os.path.join(dataset_obj_prefix,getattr(args, 'saved_cand_ids', None)),
)
if args.test:
    candidate_pool = candidate_pool[:10]

# encode in chunks to parallelize
save_file = None
if getattr(args, 'encoding_save_file_dir', None) is not None:
    save_file = os.path.join(
        gettempdir(),
        "generated_biencoder_embeddings.t7",
    )
print("Saving in: {}".format(save_file))

if save_file is not None:
    f = open(save_file, "w").close()  # mark as existing

if candidate_pool.shape[0] <2:
    candidate_encoding = encode_candidate(
        biencoder,
        candidate_pool,
        biencoder_params["encode_batch_size"],
        biencoder_params["silent"],
        logger,
    )
else:
    candidate_encoding = encode_candidate(
        biencoder,
        candidate_pool[args.chunk_start:args.chunk_end],
        biencoder_params["encode_batch_size"],
        biencoder_params["silent"],
        logger,
    )

if save_file is not None:
    torch.save(candidate_encoding, save_file)

print(candidate_encoding[0,:10])
if baseline_candidate_encoding is not None:
    print(baseline_candidate_encoding[0,:10])

dataset = Dataset.create(
            dataset_project=PROJECT_DATASET_NAME, dataset_name="generated_biencoder_embeddings"
)

dataset.add_files(os.path.join(gettempdir(), 'generated_biencoder_embeddings.t7'))
dataset.upload(output_url="s3://experiment-logging/storage")
dataset.finalize()

create_dataset(os.path.join(gettempdir(), 'generated_biencoder_embeddings.t7'), PROJECT_DATASET_NAME, OUTPUT_DATASET)


