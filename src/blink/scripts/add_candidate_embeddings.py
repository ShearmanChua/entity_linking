# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import json
import os

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--path_to_initial_embeddings', type=str, required=True, help='filepath to file containing original candidate embeddings')
parser.add_argument('--path_to_new_embeddings', type=str, required=True, help='filepath to file containing new candidate embeddings to add')
parser.add_argument('--output_dir', type=str, required=True, help='directory to save output embeddings to')
parser.add_argument('--output_filename', type=str, default='new_candidate_embeddings.t7', help='output file name')
parser.add_argument('--chunk_size', type=int, default=1000000, help='size of each chunk')
args = parser.parse_args()

CHUNK_SIZES = args.chunk_size

all_chunks = []


original_embeddings_path = args.path_to_initial_embeddings

if not os.path.exists(original_embeddings_path) or os.path.getsize(original_embeddings_path) == 0:
    print("Path to orignial embeddings incorrect or original embeddings file is empty")

print("Loading original embeddings!!!")
try:
    loaded_chunk = torch.load(original_embeddings_path)
    print("Initial number of embeddings: ",loaded_chunk.shape[0])
except:
    print("Path to orignial embeddings incorrect or unable to load torch embeddings from file path {}".format(original_embeddings_path))

all_chunks.append(loaded_chunk)

new_embeddings_path = args.path_to_new_embeddings

if not os.path.exists(new_embeddings_path) or os.path.getsize(new_embeddings_path) == 0:
    print("Path to new embeddings incorrect or original embeddings file is empty")

print("Loading new embeddings!!!")
try:
    loaded_chunk = torch.load(new_embeddings_path)
    print("Number of new embeddings to be added: ",loaded_chunk.shape[0])
except:
    print("Path to new embeddings incorrect or unable to load torch embeddings from file path {}".format(new_embeddings_path))

all_chunks.append(loaded_chunk)

all_chunks = torch.cat(all_chunks, dim=0)
torch.save(all_chunks, os.path.join(
    args.output_dir, args.output_filename
))