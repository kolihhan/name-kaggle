"""CPU-only tiny Qwen3Model API compatibility test; NO pretrained weight download."""
from __future__ import annotations
import json
import torch
from transformers import AutoModel,Qwen3Config

def main():
    config=Qwen3Config(
        vocab_size=256,hidden_size=128,intermediate_size=256,
        num_hidden_layers=8,num_attention_heads=4,num_key_value_heads=2,
        head_dim=32,max_position_embeddings=1024,
    )
    model=AutoModel.from_config(config).eval()
    ids=torch.tensor([[12,32,22,0],[12,42,73,95]],dtype=torch.long)
    mask=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.long)
    with torch.inference_mode():
        out=model(input_ids=ids,attention_mask=mask,output_hidden_states=True,
                  use_cache=False,return_dict=True)
        hs=out.hidden_states
        from scripts.ntl_decoder_probe_extract_colab_20261119 import layer_ids,masked_pooled
        layers=layer_ids(config.num_hidden_layers)
        x=masked_pooled(hs,mask,layers)
    if len(hs)!=config.num_hidden_layers+1 or x.shape!=(2,4*config.hidden_size):
        raise RuntimeError("QWEN3_HIDDEN_STATE_CONTRACT_DRIFT")
    if not torch.isfinite(x).all():raise FloatingPointError("QWEN3_BAD_FEATURES")
    print("NTL_F1_QWEN3_REAL_ARCHITECTURE_TINY_RANDOM_CPU_COMPAT="+json.dumps({
        "model":"Qwen3Model","layers":layers,"shape":list(x.shape),
        "only_random_tiny_weights":True,"pretrained_model_not_run":True,
        "gpu":False,"hidden_test":False},sort_keys=True),flush=True)

if __name__=="__main__":main()
