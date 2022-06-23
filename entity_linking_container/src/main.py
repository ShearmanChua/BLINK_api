import ast
import json
import requests
import pandas as pd
from fastapi import FastAPI, Request
from entity_linking import entity_linking
from inference import Inference

app = FastAPI()


def row_linking(row):

    data_to_link = []

    data = {
        "id": 0,
        "doc_id": row['doc_id'],
        "label": "unknown",
        "label_id": -1,
        "context_left": row['context_left'].lower() if row['context_left'] is not None else "",
        "mention": row['mention'].lower()if row['mention'] is not None else "",
        "context_right": row['context_right'].lower()if row['context_right'] is not None else ""
    }
    data_to_link.append(data)

    print(data_to_link)

    inference = Inference(data_to_link)
    results = inference.run_inference()

    if results:
        entity_id = [row['entity_id'] for row in results['entities']]
    else:
        entity_id = -1

    return entity_id


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/link")
async def link(request: Request):
    df_dict_str = await request.json()
    df_json = json.dumps(df_dict_str)
    df = pd.read_json(df_json, orient="records")
    print(df.head())
    print(df.info())

    df_linked = entity_linking(
        df,
        args={},
        model_dir="models",
        task_entity_col="entities",
        task_entity_linking_col="entities_linked",
        test_entities=None,
        test_mentions=None,
        interactive=False,
        top_k=10,
        faiss_index="hnsw",  # hnsw / flat
        biencoder_model="biencoder_wiki_large.bin",
        biencoder_config="biencoder_wiki_large.json",
        entity_catalogue="entity.jsonl",  # new_entity.jsonl
        entity_encoding="all_entities_large.t7",  # new_candidate_embeddings.t7
        crossencoder_model="crossencoder_wiki_large.bin",
        crossencoder_config="crossencoder_wiki_large.json",
        fast=False,
        output_path="logs/",
        index_path="faiss_hnsw_index.pkl",  # faiss_hnsw_index.pkl / faiss_flat_index
    )

    print(df_linked.head())
    print(df_linked.info())

    df_json = df_linked.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json

@app.post("/single_inference")
async def link(request: Request):
    dict_str = await request.json()
    json_dict = dict_str

    data = {
        "id": 0,
        "doc_id": json_dict['doc_id'] if 'doc_id' in json_dict.keys() else 0,
        "label": "unknown",
        "label_id": -1,
        "context_left": json_dict['context_left'].lower() if json_dict['context_left'] is not None else "",
        "mention": json_dict['mention'] if json_dict['mention'] is not None else "",
        "context_right": json_dict['context_right'].lower()if json_dict['context_right'] is not None else ""
    }
    
    data_to_link = []
    data_to_link.append(data)

    inference = Inference(data_to_link)
    results = inference.run_inference()

    json_string = results['entities'][0]
    json_string = json.dumps(json_string)

    return json_string

@app.post("/df_link")
async def link(request: Request):
    df_dict_str = await request.json()
    df_json = json.dumps(df_dict_str)
    df = pd.read_json(df_json, orient="records")
    print(df.head())
    print(df.info())

    data_to_link = []

    for idx, row in df.iterrows():
        data = {
            "id": idx,
            "doc_id": row['doc_id'],
            "label": "unknown",
            "label_id": -1,
            "context_left": row['context_left'].lower() if row['context_left'] is not None else "",
            "mention": row['mention'].lower()if row['mention'] is not None else "",
            "context_right": row['context_right'].lower()if row['context_right'] is not None else ""
        }

        data_to_link.append(data)

    inference = Inference(data_to_link)
    results = inference.run_inference()

    entity_ids = [row['entity_id'] for row in results['entities']]
    entity_links = [row['entity_link'] for row in results['entities']]

    print("Entity IDs length: ", len(entity_ids))

    df_linked = df
    df_linked['entity_id'] = entity_ids
    df_linked['entity_link'] = entity_links

    print(df_linked.head())
    print(df_linked.info())

    df_json = df_linked.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json

@app.post("/link_row")
async def link(request: Request):
    df_dict_str = await request.json()
    df_json = json.dumps(df_dict_str)
    df = pd.read_json(df_json, orient="records")
    print(df.head())
    print(df.info())

    entity_ids = []

    for idx, row in df.iterrows():
        entity_id = row_linking(row)
        entity_ids.append(entity_id)

    df['entity_id'] = entity_ids

    print(df.head())

    df_json = df.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json
