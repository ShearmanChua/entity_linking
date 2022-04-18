import json
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_initial_entity_jsonl', type=str, required=True, help='filepath to file containing original entities')
parser.add_argument('--path_to_entity_jsonl_to_append', type=str, required=True, help='filepath to file containing new entities to add')
parser.add_argument('--output_dir', type=str, required=True, help='directory to save output entities to')
parser.add_argument('--output_filename', type=str, default='new_entity.jsonl', help='output file name')
args = parser.parse_args()

json_list = []
with open(args.path_to_initial_entity_jsonl, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            json_list.append(entity)

print(type(json_list[0]))

new_entities_list = []
with open(args.path_to_entity_jsonl_to_append, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            new_entities_list.append(entity)


json_list.extend(new_entities_list)

with open(os.path.join(args.output_dir, args.output_filename), 'w') as outfile:
    for entry in json_list:
        json.dump(entry, outfile)
        outfile.write('\n')

print("Done adding new entities")
