import ast
import json
import requests
import pandas as pd
from fastapi import FastAPI, Request
from entity_linking import entity_linking

app = FastAPI()


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
