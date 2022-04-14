from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import argparse
import json
import torch

models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"new_entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

args = argparse.Namespace(**config)

# Load biencoder model and biencoder params just like in main_dense.py
with open(args.biencoder_config) as json_file:
    biencoder_params = json.load(json_file)
    biencoder_params["path_to_model"] = args.biencoder_model
biencoder = load_biencoder(biencoder_params)

# Read 10 entities from entity.jsonl
entities = []

print("Reading entity json file")

with open('./models/entity.jsonl') as f:
    for i, line in enumerate(f):
        entity = json.loads(line)
        entities.append(entity)

entities.append(
    {'text': 'Shearman Chua was born in Singapore, in the year 1996. He is an alumnus of NTU and is currently working at DSTA', 'idx': 'https://en.wikipedia.org/wiki?curid=88767376', 'title': 'Shearman Chua', 'entity': 'Shearman Chua'}
)

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
torch.save(ids, models_path+"new_entities_large.json")