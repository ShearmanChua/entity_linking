from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import argparse
import json
import torch

models_path = "models/" # the path where you stored the BLINK models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--biencoder_model', type=str, default="models/biencoder_wiki_large.bin", help='filepath to file containing original entities')
parser.add_argument('--biencoder_config', type=str, default="models/biencoder_wiki_large.json", help='filepath to file containing new entities to add')
parser.add_argument('--entity_catalogue', type=str, default="models/entities_to_add.jsonl", help='directory to save output entities to')
parser.add_argument('--entity_encoding', type=str, default='models/all_entities_large.t7', help='output file name')
parser.add_argument('--crossencoder_model', type=str, default='models/crossencoder_wiki_large.bin', help='output file name')
parser.add_argument('--crossencoder_config', type=str, default='models/crossencoder_wiki_large.json', help='output file name')
args = parser.parse_args()

# Load biencoder model and biencoder params just like in main_dense.py
with open(args.biencoder_config) as json_file:
    biencoder_params = json.load(json_file)
    biencoder_params["path_to_model"] = args.biencoder_model
biencoder = load_biencoder(biencoder_params)

# Read 10 entities from entity.jsonl
entities = []

print("Reading entity json file")

with open('models/entities_to_add.jsonl') as f:
    for i, line in enumerate(f):
        entity = json.loads(line)
        entities.append(entity)

print(entities)

print("Generating token_ids")

# Get token_ids corresponding to candidate title and description
tokenizer = biencoder.tokenizer
max_context_length, max_cand_length =  biencoder_params["max_context_length"], biencoder_params["max_cand_length"]
max_seq_length = max_cand_length
ids = []

for entity in entities:
    # if entity['entity'] == 'Shearman Chua':
    #     print(entity)
    #     print(entity['text'])
    #     print(entity['title'])
    candidate_desc = entity['text']
    candidate_title = entity['title']
    cand_tokens = get_candidate_representation(
        candidate_desc, 
        tokenizer, 
        max_seq_length, 
        candidate_title=candidate_title
    )

    token_ids = cand_tokens["ids"]
    ids.append(token_ids)

ids = torch.tensor(ids)
torch.save(ids, models_path+"new_entities_ids.json")