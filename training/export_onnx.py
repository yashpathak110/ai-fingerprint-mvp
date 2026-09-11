# training/export_onnx.py
import torch, yaml
from unsloth import FastLanguageModel
import argparse, os

def export_onnx(config):
    cfg = config['training']
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg['output_dir'], # Load fine-tuned
        max_seq_length=512, dtype=None, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    
    dummy = torch.ones(1, 512, dtype=torch.long)
    torch.onnx.export(model, (dummy,), cfg['onnx_path'], 
                      input_names=['input_ids'], output_names=['logits'],
                      dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}},
                      opset_version=17)
    print(f"Exported to {cfg['onnx_path']}")

if __name__ == "__main__":
    with open("config.yaml") as f: config = yaml.safe_load(f)
    export_onnx(config)
