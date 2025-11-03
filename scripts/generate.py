import os
import argparse
import logging
import sys

import torch

from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.model import Transformer
from cse599o_basics.utils import load_checkpoint
from cse599o_basics.calc import softmax


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)

logger.addHandler(handler)


def apply_top_p_sampling(probs, top_p):
    if top_p >= 1.0:
        return probs
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    
    return probs / probs.sum()


def main(args):
    dtype_str = args.dtype
    d_model = args.dim
    n_layers = args.n_layers
    n_heads = args.n_heads
    rope_theta = args.rope_theta
    d_ff = args.ffn_dim
    context_length = args.context_length
    checkpoint_load_path = args.checkpoint_load_path
    prompt_path = args.prompt_path
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p

    eof_str = "<|endoftext|>"
    tokenizer = BPETokenizer(None, None, [eof_str])
    eof_tokens = tokenizer.encode(eof_str)

    prompt_tokens = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            prompt_tokens.extend(tokenizer.encode(line.strip()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[dtype_str]
    torch.set_default_dtype(dtype)
    logger.info(f"Initializing model on device {device} with dtype {torch.get_default_dtype()}")
    
    vocab_size  = tokenizer.tokenizer.n_vocab
    
    model = Transformer(
        d_model,
        n_heads,
        d_ff,
        vocab_size,
        context_length,
        n_layers,
        rope_theta,
        dtype=dtype,
    ).to(device)
    load_checkpoint(checkpoint_load_path, model)
    model.eval()

    generated_token = -1
    while True:
        input_tokens = prompt_tokens[-context_length:]
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            logit = model(input_ids)[0, -1, :]
            probs = softmax(logit / temperature, -1)
            probs = apply_top_p_sampling(probs, top_p)
            generated_token = torch.multinomial(probs, num_samples=1).item()

        prompt_tokens.append(generated_token)

        if len(prompt_tokens) >= len(eof_tokens) and prompt_tokens[-len(eof_tokens):] == eof_tokens:
            break

        if len(prompt_tokens) >= max_tokens:
            break

        decoded_text = tokenizer.decode(prompt_tokens)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(decoded_text)

    logger.info(f"Generation complete, generated {len(prompt_tokens)} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer model")

    parser.add_argument("--checkpoint-load-path", type=str, required=True, help="Path to load a checkpoint before training resumes")
    parser.add_argument("--prompt-path", type=str, required=True, help="Path to initial text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate (default: 1024)")
    
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"], help="Model data type. One of: float32, bfloat16, or float16 (default: float32)")
    parser.add_argument("--dim", type=int, required=True, help="Model hidden dimension size")
    parser.add_argument("--n-layers", type=int, required=True, help="Number of Transformer blocks")
    parser.add_argument("--n-heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--rope-theta", type=int, required=True, help="Base frequency for rotary positional embeddings")
    parser.add_argument("--ffn-dim", type=int, required=True, help="Model feed-forward layer dimension size")
    # parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size of dataset")
    parser.add_argument("--context-length", type=int, required=True, help="Maximum sequence length")

    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p value (default: 1.0)")
    
    args = parser.parse_args()
    main(args)
