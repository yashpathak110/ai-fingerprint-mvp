# vector_db/ingest.py
import yaml, pandas as pd
from client import QdrantManager

def ingest(config):
    mgr = QdrantManager(config)
    df = pd.read_parquet(config['data']['processed_dir'] + "/" + config['data']['train_file'])
    texts = df['text'].tolist()
    payloads = df[['source_model', 'prompt_type']].to_dict('records')
    mgr.upsert_corpus(texts, payloads)
    print(f"Ingested {len(texts)} vectors.")

if __name__ == "__main__":
    with open("../config.yaml") as f: config = yaml.safe_load(f) # Note: ../
    ingest(config)
