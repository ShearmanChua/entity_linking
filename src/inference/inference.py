import blink.main_dense as main_dense
import blink.ner as NER
from blink.crossencoder.train_cross import modify
from blink.crossencoder.data_process import prepare_crossencoder_data
import argparse
import time
import ast

models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"new_entity.jsonl",
    "entity_encoding": models_path+"new_candidate_embeddings.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/", # logging directory
    "faiss_index": "hnsw",
    "index_path": "models/faiss_index.pkl"
}

args = argparse.Namespace(**config)

def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer=None,
    test_data=None,
):

    if not test_data and not args.test_mentions and not args.interactive:
        msg = (
            "ERROR: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entitied (--test_entities)"
        )
        raise ValueError(msg)

    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

    stopping_condition = False
    while not stopping_condition:

        samples = None

        if args.interactive:
            logger.info("interactive mode")

            # biencoder_params["eval_batch_size"] = 1

            # Load NER model
            ner_model = NER.get_model()

            # Interactive
            text = input("insert text:")

            # Identify mentions
            samples = main_dense._annotate(ner_model, [text])

            main_dense._print_colorful_text(text, samples)

        else:
            if logger:
                logger.info("test dataset mode")

            if test_data:
                samples = test_data
            else:
                # Load test mentions
                samples = main_dense._get_test_samples(
                    args.test_mentions,
                    args.test_entities,
                    title2id,
                    wikipedia_id2local_id,
                    logger,
                )

            stopping_condition = True

        # don't look at labels
        keep_all = (
            args.interactive
            or samples[0]["label"] == "unknown"
            or samples[0]["label_id"] < 0
        )

        # prepare the data for biencoder
        if logger:
            logger.info("preparing data for biencoder")
        dataloader = main_dense._process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        # run biencoder
        if logger:
            logger.info("run biencoder")
        top_k = args.top_k
        labels, nns, scores = main_dense._run_biencoder(
            biencoder, dataloader, candidate_encoding, top_k, faiss_indexer
        )

        if args.interactive:

            print("\nfast (biencoder) predictions:")

            main_dense._print_colorful_text(text, samples)

            # print biencoder prediction
            idx = 0
            for entity_list, sample in zip(nns, samples):
                e_id = entity_list[0]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                main_dense._print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()

            if args.fast:
                # use only biencoder
                continue

        else:

            biencoder_accuracy = -1
            recall_at = -1
            if not keep_all:
                # get recall values
                top_k = args.top_k
                x = []
                y = []
                for i in range(1, top_k):
                    temp_y = 0.0
                    for label, top in zip(labels, nns):
                        if label in top[:i]:
                            temp_y += 1
                    if len(labels) > 0:
                        temp_y /= len(labels)
                    x.append(i)
                    y.append(temp_y)
                # plt.plot(x, y)
                biencoder_accuracy = y[0]
                recall_at = y[-1]
                print("biencoder accuracy: %.4f" % biencoder_accuracy)
                print("biencoder recall@%d: %.4f" % (top_k, y[-1]))

            if args.fast:

                predictions = []
                for entity_list in nns:
                    sample_prediction = []
                    for e_id in entity_list:
                        e_title = id2title[e_id]
                        sample_prediction.append(e_title)
                    predictions.append(sample_prediction)

                # use only biencoder
                return (
                    biencoder_accuracy,
                    recall_at,
                    -1,
                    -1,
                    len(samples),
                    predictions,
                    scores,
                )

        # prepare crossencoder data
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            crossencoder.tokenizer, samples, labels, nns, id2title, id2text, keep_all,
        )

        context_input = modify(
            context_input, candidate_input, crossencoder_params["max_seq_length"]
        )

        dataloader = main_dense._process_crossencoder_dataloader(
            context_input, label_input, crossencoder_params
        )

        # run crossencoder and get accuracy
        accuracy, index_array, unsorted_scores = main_dense._run_crossencoder(
            crossencoder,
            dataloader,
            logger,
            context_len=biencoder_params["max_context_length"],
        )

        if args.interactive:

            print("\naccurate (crossencoder) predictions:")

            main_dense._print_colorful_text(text, samples)

            # print crossencoder prediction
            idx = 0
            for entity_list, index_list, sample in zip(nns, index_array, samples):
                e_id = entity_list[index_list[-1]]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                main_dense._print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()
        else:

            scores = []
            predictions = []
            links = []
            for entity_list, index_list, scores_list in zip(
                nns, index_array, unsorted_scores
            ):

                index_list = index_list.tolist()

                # descending order
                index_list.reverse()

                sample_prediction = []
                sample_links = []
                sample_scores = []
                for index in index_list:
                    e_id = entity_list[index]
                    e_title = id2title[e_id]
                    e_url = id2url[e_id]
                    sample_prediction.append(e_title)
                    sample_links.append(e_url)
                    sample_scores.append(scores_list[index])
                predictions.append(sample_prediction)
                scores.append(sample_scores)
                links.append(sample_links)

            crossencoder_normalized_accuracy = -1
            overall_unormalized_accuracy = -1
            if not keep_all:
                crossencoder_normalized_accuracy = accuracy
                print(
                    "crossencoder normalized accuracy: %.4f"
                    % crossencoder_normalized_accuracy
                )

                if len(samples) > 0:
                    overall_unormalized_accuracy = (
                        crossencoder_normalized_accuracy * len(label_input) / len(samples)
                    )
                print(
                    "overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy
                )
            return (
                biencoder_accuracy,
                recall_at,
                crossencoder_normalized_accuracy,
                overall_unormalized_accuracy,
                len(samples),
                predictions,
                links,
                scores,
            )

main_dense.run = run

def inferenceWrapper(cls):
      
    class Wrapper:
          
        def __init__(self, mentions_to_link):

            self.args = args

            start = time.time()
            self.models = main_dense.load_models(args, logger=None)
            end = time.time()

            print("Time to load BLINK models",end - start)

            self.wrap = cls(mentions_to_link)
              
        def run_inference(self):

            start = time.time()

            if len(self.wrap.mentions_to_link) >0:
                try:
                    _, _, _, _, _, predictions, links, scores = main_dense.run(self.args, None, *self.models, test_data=self.wrap.mentions_to_link)
                    error = False
                except:
                    print("Error while performing entity linking on mentions list")
                    error = True
            else:
                error = True
            
            end = time.time()
            print("Time to complete entity linking",end - start)

            entities_dict ={}
            if not error:
                ent_list = []
                identified_entities = []
                for i in range(0,len(self.wrap.mentions_to_link)):
                    ent_dict = {}
                    print("Mention: ",self.wrap.mentions_to_link[i]["mention"])
                    print("Original sentence: ")
                    print(self.wrap.mentions_to_link[i]["context_left"] + " " + self.wrap.mentions_to_link[i]["mention"] + " " + self.wrap.mentions_to_link[i]["context_right"])
                    print("Entity linked: ", predictions[i][0])
                    print("Score: ", scores[i][0])
                    print("\n")

                    if scores[i][0] > 0:
                        
                        if predictions[i][0] not in identified_entities: 

                            ent_dict['text'] = self.wrap.mentions_to_link[i]["context_left"] + " " + self.wrap.mentions_to_link[i]["mention"] + " " + self.wrap.mentions_to_link[i]["context_right"]
                            ent_dict['mention'] = self.wrap.mentions_to_link[i]["mention"]
                            ent_dict['entity_linked'] = predictions[i][0]
                            ent_dict['entity_id'] = links[i][0]
                            ent_dict['entity_confidence_score'] = scores[i][0]
                            identified_entities.append(predictions[i][0])
                    
                    ent_list.append(ent_dict)
                entities_dict["identified_entities"] = identified_entities
                entities_dict['entities'] = ent_list

            return entities_dict
          
    return Wrapper
  
@inferenceWrapper
class Inference:
    def __init__(self, mentions_to_link):

        # mentions_to_link = ast.literal_eval(mentions_to_link)

        for mention in mentions_to_link:
            if "label" not in mention.keys():
                mention["label"] = "unknown"
            if "label" not in mention.keys():
                mention["label_id"] = -1
            if not (set(["context_left","mention","context_right"]).issubset(set(list(mention.keys())))):
                print("Mention dictionary does not contain 'context_left','mention', or 'context_right' field, will result in error when running inference.")

        self.mentions_to_link = mentions_to_link