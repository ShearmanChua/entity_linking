import json

json_list = []
with open('models/entity.jsonl', "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            json_list.append(entity)

# with open('models/entity.jsonl', 'r') as json_file:
#     json_list = list(json_file)

print(type(json_list[0]))

json_list.append(
    {'text': 'Shearman Chua was born in Singapore, in the year 1996. He is an alumnus of NTU and is currently working at DSTA', 'idx': 'https://en.wikipedia.org/wiki?curid=88767376', 'title': 'Shearman Chua', 'entity': 'Shearman Chua'}
)

with open("models/new_entity.jsonl", 'w') as outfile:
    for entry in json_list:
        json.dump(entry, outfile)
        outfile.write('\n')

print("Done adding new entities")
