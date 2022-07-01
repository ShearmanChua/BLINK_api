import blink.main_dense as main_dense
import blink.ner as NER
import blink.candidate_ranking.utils as utils
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
from blink.crossencoder.train_cross import modify
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)

import argparse
import pathlib
import time
import ast
import json
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import numpy as np

model_dir = "models" # the path where you stored the BLINK models

config = {}

absolute_path = pathlib.Path(__file__).parent.resolve()

args = argparse.Namespace(**config)

args.biencoder_model = os.path.join(absolute_path, model_dir, "biencoder_wiki_large.bin")
args.biencoder_config = os.path.join(absolute_path, model_dir, "biencoder_wiki_large.json")
args.entity_catalogue = os.path.join(absolute_path, model_dir, "entity.jsonl")
args.entity_encoding = os.path.join(absolute_path, model_dir, "all_entities_large.t7")
args.crossencoder_model = os.path.join(absolute_path, model_dir, "crossencoder_wiki_large.bin")
args.crossencoder_config = os.path.join(
    absolute_path, model_dir, "crossencoder_wiki_large.json"
)
args.entities_to_add = os.path.join(absolute_path, model_dir,"entities_to_add.jsonl")
args.new_entity_catalogue = os.path.join(absolute_path, model_dir, "entity.jsonl")
args.index_path = os.path.join(absolute_path, model_dir, "faiss_index.pkl")

args.fast = False
args.output_path = "logs/"
args.test_entities = None
args.test_mentions = None
args.interactive = False
args.top_k = 10
args.faiss_index = "hnsw"

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
            ids = []
            for entity_list, index_list, scores_list in zip(
                nns, index_array, unsorted_scores
            ):

                index_list = index_list.tolist()

                # descending order
                index_list.reverse()

                sample_prediction = []
                sample_links = []
                sample_ids = []
                sample_scores = []
                for index in index_list:
                    e_id = entity_list[index]
                    e_title = id2title[e_id]
                    e_url = id2url[e_id]
                    sample_prediction.append(e_title)
                    sample_links.append(e_url)
                    sample_ids.append(int(e_id))
                    sample_scores.append(float(scores_list[index]))
                predictions.append(sample_prediction)
                scores.append(sample_scores)
                links.append(sample_links)
                ids.append(sample_ids)

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
                ids,
                scores,
            )

def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_input=context_input.cuda()
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores

main_dense.run = run
# main_dense._run_biencoder = _run_biencoder

def inferenceWrapper(cls):
      
    class Wrapper:
          
        def __init__(self, mentions_to_link):

            self.args = args

            start = time.time()
            self.models = main_dense.load_models(args, logger=None)
            end = time.time()

            print("Time to load BLINK models",end - start)

            # load entities
            self.title2id = {}
            self.id2title = {}
            self.id2text = {}
            self.wikipedia_id2local_id = {}
            local_idx = 0
            with open(self.args.entity_catalogue, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    entity = json.loads(line)

                    if "idx" in entity:
                        split = entity["idx"].split("curid=")
                        if len(split) > 1:
                            wikipedia_id = int(split[-1].strip())
                        else:
                            wikipedia_id = entity["idx"].strip()

                        assert wikipedia_id not in self.wikipedia_id2local_id
                        self.wikipedia_id2local_id[wikipedia_id] = local_idx

                    self.title2id[entity["title"].lower()] = local_idx
                    self.id2title[local_idx] = entity["title"]
                    self.id2text[local_idx] = entity["text"]
                    local_idx += 1

            self.id2url = {
                v: "https://en.wikipedia.org/wiki?curid=%s" % k
                for k, v in self.wikipedia_id2local_id.items()
            }

            self.wrap = cls(mentions_to_link)
              
        def run_inference(self):

            start = time.time()

            if len(self.wrap.mentions_to_link) >0:

                mentions_to_blink = {
                    "ids":[],
                    "mentions":[]
                }
                mentions_to_exact_match = {
                    "ids":[],
                    "mentions":[]
                }

                for mention in self.wrap.mentions_to_link:
                    mention_length = mention["mention"].split(' ')
                    mention_length = [part for part in mention_length if len(part)>0]
                    if mention["mention"].lower() in self.title2id.keys() and len(mention_length) > 1:
                        print("Exact matched: ", mention["mention"])
                        mentions_to_exact_match["ids"].append(mention["id"])
                        mention["entity_linked"] = mention["mention"].title()
                        mention["entity_id"] = self.title2id[mention["mention"].lower()]
                        mention["entity_link"] = self.id2url[mention["entity_id"]]
                        mentions_to_exact_match["mentions"].append(mention)
                    else:
                        print("To BLINK infer: ", mention["mention"])
                        mentions_to_blink["ids"].append(mention["id"])
                        mentions_to_blink["mentions"].append(mention)

                try:
                    _, _, _, _, _, predictions, links, ids,scores = main_dense.run(self.args, None, *self.models, test_data=mentions_to_blink["mentions"])
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
                for i in range(0,len(self.wrap.mentions_to_link)):
                    ent_dict = {}
                    print("Mention: ",self.wrap.mentions_to_link[i]["mention"])
                    print("Original sentence: ")
                    print(self.wrap.mentions_to_link[i]["context_left"] + " " + self.wrap.mentions_to_link[i]["mention"] + " " + self.wrap.mentions_to_link[i]["context_right"])

                    if self.wrap.mentions_to_link[i]["id"] in mentions_to_blink["ids"]:
                        index = mentions_to_blink["ids"].index(self.wrap.mentions_to_link[i]["id"])
                        print("Entity linked: ", predictions[index][0])
                        print("Entities identified: ", predictions[index])
                        print("Score: ", scores[index][0])
                        print("\n")

                        if scores[index][0] > -2.0:

                            ent_dict['doc_id'] = self.wrap.mentions_to_link[i]["doc_id"]
                            ent_dict['mention'] = self.wrap.mentions_to_link[i]["mention"]
                            ent_dict['entity_linked'] = predictions[index][0]
                            ent_dict['entity_link'] = links[index][0]
                            ent_dict['entity_id'] = ids[index][0]
                            ent_dict['entity_confidence_score'] = scores[index][0]

                        else:
                            ent_dict['doc_id'] = self.wrap.mentions_to_link[i]["doc_id"]
                            ent_dict['mention'] = self.wrap.mentions_to_link[i]["mention"]
                            ent_dict['entity_linked'] = ""
                            ent_dict['entity_link'] = ""
                            ent_dict['entity_id'] = -1
                            ent_dict['entity_confidence_score'] = 0.0
                        
                        ent_list.append(ent_dict)
                    else:
                        index = mentions_to_exact_match["ids"].index(self.wrap.mentions_to_link[i]["id"])
                        ent_dict['doc_id'] = self.wrap.mentions_to_link[i]["doc_id"]
                        ent_dict['mention'] = self.wrap.mentions_to_link[i]["mention"]
                        ent_dict['entity_linked'] = mentions_to_exact_match["mentions"][index]['entity_linked']
                        ent_dict['entity_link'] = mentions_to_exact_match["mentions"][index]['entity_link']
                        ent_dict['entity_id'] = mentions_to_exact_match["mentions"][index]['entity_id']
                        ent_dict['entity_confidence_score'] = 1.0
                        ent_list.append(ent_dict)

                entities_dict['entities'] = ent_list

            return entities_dict
          
    return Wrapper

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

def KBWrapper(cls):
      
    class Wrapper:
          
        def __init__(self, entities_to_add):

            self.args = args

            self.wrap = cls(entities_to_add)

        def add_to_jsonl_kb(self,new_entities_list):

            json_list = []
            with open(self.args.entity_catalogue, "r") as fin:
                    lines = fin.readlines()
                    for line in lines:
                        entity = json.loads(line)
                        json_list.append(entity)

            json_list.extend(new_entities_list)

            with open(self.args.new_entity_catalogue, 'w') as outfile:
                for entry in json_list:
                    json.dump(entry, outfile)
                    outfile.write('\n')

            print("Done adding new entities")

            return new_entities_list

        def generate_biencoder_token_ids(self,entities):

            with open(self.args.biencoder_config) as json_file:
                biencoder_params = json.load(json_file)
                biencoder_params["path_to_model"] = self.args.biencoder_model
            biencoder = load_biencoder(biencoder_params)

            print(entities)

            print("Generating token_ids")

            # Get token_ids corresponding to candidate title and description
            tokenizer = biencoder.tokenizer
            max_context_length, max_cand_length =  biencoder_params["max_context_length"], biencoder_params["max_cand_length"]
            max_seq_length = max_cand_length
            ids = []

            for entity in entities:
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

            return ids

        def generate_candidates(self,biencoder_ids):

            with open(self.args.biencoder_config) as json_file:
                biencoder_params = json.load(json_file)
                biencoder_params["path_to_model"] = self.args.biencoder_model
            
            biencoder_params["entity_dict_path"] = self.args.entities_to_add
            biencoder_params["data_parallel"] = True
            biencoder_params["no_cuda"] = False
            biencoder_params["max_context_length"] = 32
            biencoder_params["encode_batch_size"] = 8

            biencoder = load_biencoder(biencoder_params)

            logger = utils.get_logger(biencoder_params.get("model_output_path", None))

            # candidate_pool = load_candidate_pool(
            #     biencoder.tokenizer,
            #     biencoder_params,
            #     logger,
            #     getattr(args, 'saved_cand_ids', None),
            # )

            candidate_pool = biencoder_ids

            print(candidate_pool.shape)
      
            candidate_encoding = encode_candidate(
                biencoder,
                candidate_pool,
                biencoder_params["encode_batch_size"],
                biencoder_params["silent"],
                logger,
            )
            
            print(candidate_encoding.shape)

            print(candidate_encoding[0,:10])

            return candidate_encoding

        def merge_with_original_embeddings(self,candidate_encoding):
            all_chunks = []

            original_embeddings_path = self.args.entity_encoding

            if not os.path.exists(original_embeddings_path) or os.path.getsize(original_embeddings_path) == 0:
                print("Path to orignial embeddings incorrect or original embeddings file is empty")

            print("Loading original embeddings!!!")

            try:
                loaded_chunk = torch.load(original_embeddings_path)
                print("Initial number of embeddings: ",loaded_chunk.shape[0])
            except:
                print("Path to orignial embeddings incorrect or unable to load torch embeddings from file path {}".format(original_embeddings_path))

            all_chunks.append(loaded_chunk)

            all_chunks.append(candidate_encoding)

            all_chunks = torch.cat(all_chunks, dim=0)

            print(all_chunks.shape)

            del loaded_chunk

            torch.save(all_chunks, 'models/test_candidate_embeddings.t7')

            print("Saved in 'models/test_candidate_embeddings.t7' ")

            return all_chunks

        def create_faiss_index(self,candidate_encoding):
            output_path = self.args.faiss_output_path
            output_dir, _ = os.path.split(output_path)
            logger = utils.get_logger(output_dir)

            vector_size = candidate_encoding.size(1)
            index_buffer = 50000

            if True:
                logger.info("Using HNSW index in FAISS")
                index = DenseHNSWFlatIndexer(vector_size, index_buffer)
            else:
                logger.info("Using Flat index in FAISS")
                index = DenseFlatIndexer(vector_size, index_buffer)

            logger.info("Building index.")
            index.index_data(candidate_encoding.numpy())
            logger.info("Done indexing data.")

            index.serialize(output_path)

            return

        def add_entities_to_kb(self):

            print("Runnning add_to_jsonl_kb!!!")
            new_entities = self.add_to_jsonl_kb(self.wrap.entities_to_add)
            torch.cuda.empty_cache()
            print("Runnning generate_biencoder_token_ids!!!")
            biencoder_ids = self.generate_biencoder_token_ids(new_entities)
            torch.cuda.empty_cache()
            print("Runnning generate_candidates!!!")
            candidate_embedding = self.generate_candidates(biencoder_ids)
            torch.cuda.empty_cache()
            print("Runnning merge_with_original_embeddings!!!")
            full_candidate_embedding = self.merge_with_original_embeddings(candidate_embedding)
            torch.cuda.empty_cache()
            print("Runnning create_faiss_index!!!")
            self.create_faiss_index(full_candidate_embedding)
            print("Done!")

            return
          
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

@KBWrapper
class NewKBEntities:
    def __init__(self, entities_to_add):

        
        # [{"text": " Shearman Chua was born in Singapore, in the year 1996. He is an alumnus of NTU and is currently working at DSTA. ", "idx": "https://en.wikipedia.org/wiki?curid=88767376", "title": "Shearman Chua", "entity": "Shearman Chua"},
        # {"text": " The COVID-19 recession is a global economic recession caused by the COVID-19 pandemic. The recession began in most countries in February 2020. After a year of global economic slowdown that saw stagnation of economic growth and consumer activity, the COVID-19 lockdowns and other precautions taken in early 2020 drove the global economy into crisis. Within seven months, every advanced economy had fallen to recession. The first major sign of recession was the 2020 stock market crash, which saw major indices drop 20 to 30% in late February and March. Recovery began in early April 2020, as of April 2022, the GDP for most major economies has either returned to or exceeded pre-pandemic levels and many market indices recovered or even set new records by late 2020. ", "idx": "https://en.wikipedia.org/wiki?curid=63462234", "title": "COVID-19 recession", "entity": "COVID-19 recession"},
        # {"text": " The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The novel virus was first identified from an outbreak in Wuhan, China, in December 2019. Attempts to contain it there failed, allowing the virus to spread worldwide. The World Health Organization (WHO) declared a Public Health Emergency of International Concern on 30 January 2020 and a pandemic on 11 March 2020. As of 15 April 2022, the pandemic had caused more than 502 million cases and 6.19 million deaths, making it one of the deadliest in history. ", "idx": "https://en.wikipedia.org/wiki?curid=62750956", "title": "COVID-19 pandemic", "entity": "COVID-19 pandemic"}]

        self.entities_to_add = entities_to_add