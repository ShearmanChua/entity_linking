import blink.main_dense as main_dense
import argparse
import time
import json
import torch
import src.inference.inference as inference

# models_path = "models/" # the path where you stored the BLINK models

# config = {
#     "test_entities": None,
#     "test_mentions": None,
#     "interactive": False,
#     "top_k": 10,
#     "biencoder_model": models_path+"biencoder_wiki_large.bin",
#     "biencoder_config": models_path+"biencoder_wiki_large.json",
#     "entity_catalogue": models_path+"new_entity.jsonl",
#     "entity_encoding": models_path+"new_candidate_embeddings.t7",
#     "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
#     "crossencoder_config": models_path+"crossencoder_wiki_large.json",
#     "fast": False, # set this to be true if speed is a concern
#     "output_path": "logs/", # logging directory
#     "faiss_index": "hnsw",
#     "index_path": "models/faiss_index.pkl"
# }

# args = argparse.Namespace(**config)

# with open(args.biencoder_config) as json_file:
#     biencoder_params = json.load(json_file)

# start = time.time()
# models = main_dense.load_models(args, logger=None)
# end = time.time()
# print("Time to load BLINK models",end - start)

data_to_link = [ {
                    "id": 0,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Shakespeare".lower(),
                    "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
                },
                {
                    "id": 1,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Shearman Chua".lower(),
                    "context_right": "wgo recently graduated is currently working at DSTA".lower(),
                },
                {
                    "id": 2,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Born in British Singapore,".lower(),
                    "mention": "Lee".lower(),
                    "context_right": "is the eldest son of Singapore's first prime minister, Lee Kuan Yew.".lower(),
                },
                {
                    "id": 3,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Lim Chin Siong".lower(),
                    "context_right": "was a Singaporean politician and trade union leader active in Singapore in the 1950s and 1960s. He was one of the founders of the governing People's Action Party (PAP) in 1954 when he used his popularity to galvanise many trade unions in support of the PAP.".lower(),
                },
                {
                    "id": 4,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": "Obama".lower(),
                    "context_right": "has also gone kite-surfing with billionaire Richard Branson, spoken with young leaders in Chicago and paid a visit to his home state of Hawaii.".lower(),
                },
                {
                    "id": 5,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "While COVID-related requirements vary widely across China, many localities have been taking increasingly cautious approaches recently. Dong said that his order numbers and earnings had dropped by half since the start of March due to the impact of the".lower(),
                    "mention": "COVID-19".lower(),
                    "context_right": "policies he had encountered.".lower(),
                }
                ]

# start = time.time()
# _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
# end = time.time()
# print("Time to complete entity linking",end - start)

# print(predictions, scores)

# for i in range(0,len(data_to_link)):
#     print("Original sentence: ")
#     print(data_to_link[i]["context_left"] + " " + data_to_link[i]["mention"] + " " + data_to_link[i]["context_right"])
#     print("Entity linked: ", predictions[i][0])
#     print("Score: ", scores[i][0])
#     print("\n")


inference = inference.Inference(data_to_link)
results = inference.run_inference()
print(results)
