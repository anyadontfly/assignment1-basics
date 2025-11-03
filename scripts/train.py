import os
import argparse
import logging
import sys
import csv

import numpy as np
import torch

from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.model import Transformer
from cse599o_basics.utils import (
    data_loading,
    save_checkpoint,
    load_checkpoint,
    AdamW,
    cross_entropy_loss,
    learning_rate_schedule,
    gradient_clipping,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)

logger.addHandler(handler)


def load_or_encode_data(path, tokenizer) -> np.memmap:
    vocab_size = tokenizer.tokenizer.n_vocab
    dtype_np = np.uint16 if vocab_size <= 65535 else np.uint32

    if path.endswith(".txt"):
        bin_path = os.path.splitext(path)[0] + ".bin"
        logger.info(f"Encoding text file {path} to {bin_path}")

        with open(path, "r", encoding="utf-8") as f, open(bin_path, "wb") as out:
            for line in f:
                ids = np.array(tokenizer.encode(line.strip()), dtype=dtype_np)
                bad_ids = ids[ids > 50256]
                if bad_ids.size > 0:
                    print(f"Found invalid token IDs (>50256): {ids}, result from {line.strip()}")
                ids.tofile(out)

        path = bin_path
    elif path.endswith(".bin"):
        logger.info(f"Using pre-tokenized binary file: {path}")
    else:
        raise ValueError("Unsupported file type: expected .txt or .bin")

    train_data = np.memmap(path, dtype=dtype_np, mode="r")
    return train_data

def evaluate_per_token_loss(
    model,
    valid_data,
    batch_size,
    context_length,
    vocab_size,
    device,
    num_batches=100
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = data_loading(valid_data, batch_size, context_length, device)
            logits = model(inputs)
            loss = cross_entropy_loss(logits.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item() * batch_size * context_length
            total_tokens += batch_size * context_length

    avg_loss = total_loss / total_tokens
    model.train()
    return avg_loss

def initialize_loss_csv(loss_log_path):
    if not loss_log_path:
        return None, None
    
    os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
    
    file_exists = os.path.exists(loss_log_path)
    loss_csv_file = open(loss_log_path, 'a', newline='')
    loss_csv_writer = csv.writer(loss_csv_file)
    
    if not file_exists:
        loss_csv_writer.writerow(['step', 'train_loss', 'valid_loss', 'learning_rate'])
        loss_csv_file.flush()
    
    logger.info(f"Logging losses to {loss_log_path}")
    
    return loss_csv_file, loss_csv_writer

def main(args):
    dtype_str   = args.dtype
    d_model     = args.dim
    n_layers    = args.n_layers
    n_heads     = args.n_heads
    rope_theta  = args.rope_theta
    d_ff        = args.ffn_dim
    context_length = args.context_length

    iters                       = args.iters
    lr                          = args.lr
    use_learning_rate_schedule  = args.use_learning_rate_schedule
    lr_max                      = args.lr_max
    lr_min                      = args.lr_min
    warm_up_iters               = args.warm_up_iters
    cosine_annealing_iters      = args.cosine_annealing_iters
    weight_decay                = args.weight_decay
    betas                       = (args.beta1, args.beta2)
    eps                         = args.eps
    use_grad_clip               = args.use_gradient_clipping
    max_norm_grad_clip          = args.max_norm_gradient_clipping
    eps_grad_clip               = args.eps_gradient_clipping

    batch_size                  = args.batch_size
    train_data_path             = args.train_data_path
    valid_data_path             = args.valid_data_path
    checkpoint_load_path        = args.checkpoint_load_path
    checkpoint_iters            = args.checkpoint_iters
    checkpoint_save_path        = args.checkpoint_save_path
    loss_log_path               = args.loss_log_path

    tokenizer = BPETokenizer(None, None, ["<|endoftext|>"])
    train_data = load_or_encode_data(train_data_path, tokenizer)
    valid_data = load_or_encode_data(valid_data_path, tokenizer)

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
    optim = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    step = 1
    if checkpoint_load_path and os.path.exists(checkpoint_load_path):
        step = load_checkpoint(checkpoint_load_path, model, optim)
        if iters < step:
            raise ValueError(
                f"Total training iterations ({iters}) must be greater than the checkpoint's current step ({step})"
            )
        step += 1

    logger.info(f"Starting training at step {step}/{iters} on device={device}")
    model.train()

    loss_csv_file, loss_csv_writer = initialize_loss_csv(loss_log_path)

    while step <= iters:
        if use_learning_rate_schedule:
            lr_t = learning_rate_schedule(step, lr_max, lr_min, warm_up_iters, cosine_annealing_iters)
            for group in optim.param_groups:
                group["lr"] = lr_t

        inputs, targets = data_loading(train_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits.view(-1, vocab_size), targets.view(-1))

        optim.zero_grad()
        loss.backward()
        if use_grad_clip:
            gradient_clipping(model.parameters(), max_norm_grad_clip, eps=eps_grad_clip)
        optim.step()
        current_lr = optim.param_groups[0]["lr"]
        if use_learning_rate_schedule:
            logger.info(f"Step {step}/{iters} | Loss: {loss.item():.4f} | Lr scheduled: {lr_t:.6f}")
        else:
            logger.info(f"Step {step}/{iters} | Loss: {loss.item():.4f}")

        valid_loss = None
        if (step % 100) == 0:
            valid_loss = evaluate_per_token_loss(
                model,
                valid_data,
                batch_size * 2,
                context_length,
                vocab_size,
                device,
            )
            logger.info(f"Step {step} validation loss: {valid_loss:.4f}")
        
        if loss_csv_writer:
            loss_csv_writer.writerow([step, f"{loss.item():.6f}", f"{valid_loss:.6f}" if valid_loss else "", f"{current_lr:.8f}"])
            loss_csv_file.flush()

        if (step % checkpoint_iters) == 0:
            if checkpoint_save_path is not None and checkpoint_iters is not None:
                logger.info(f"Saving checkpoint at step {step} to {checkpoint_save_path}")
                save_checkpoint(model, optim, step, checkpoint_save_path)

        step += 1

    if loss_csv_file:
        loss_csv_file.close()
        logger.info(f"Loss log saved to {loss_log_path}")
    
    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer model")

    parser.add_argument("--batch-size", type=int, required=True, help="Training batch size")
    parser.add_argument("--train-data-path", type=str, required=True, help="Path to the training text file or pre-tokenized dataset")
    parser.add_argument("--valid-data-path", type=str, required=True, help="Path to the validation text file or pre-tokenized dataset")
    parser.add_argument("--checkpoint-save-path", type=str, default=None, help="Optional path to save model checkpoints")
    parser.add_argument("--checkpoint-iters", type=int, default=None, help="Number of iterations between saving checkpoints (required if --checkpoint-save-path is set)")
    parser.add_argument("--checkpoint-load-path", type=str, default=None, help="Optional path to load a checkpoint before training resumes")
    parser.add_argument("--loss-log-path", type=str, default=None, help="Optional path to save training and validation losses as CSV for plotting learning curves")
    
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"], help="Model data type. One of: float32, bfloat16, or float16 (default: float32)")
    parser.add_argument("--dim", type=int, required=True, help="Model hidden dimension size")
    parser.add_argument("--n-layers", type=int, required=True, help="Number of Transformer blocks")
    parser.add_argument("--n-heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--rope-theta", type=int, required=True, help="Base frequency for rotary positional embeddings")
    parser.add_argument("--ffn-dim", type=int, required=True, help="Model feed-forward layer dimension size")
    # parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size of dataset")
    parser.add_argument("--context-length", type=int, required=True, help="Maximum sequence length")

    parser.add_argument("--iters", type=int, required=True, help="Total number of training iterations")
    parser.add_argument("--optim", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer name")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay, default to 0.0")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 (first momentum) for optimizer, default to 0.9")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 (second momentum) for optimizer, default to 0.999")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon value for optimizer stability, default to 1e-8")

    parser.add_argument("--use-learning-rate-schedule", type=bool, default=False, help="Whether to use learning rate schedule")
    parser.add_argument("--lr-max", type=float, default=None, help="Maximum learning rate")
    parser.add_argument("--lr-min", type=float, default=None, help="Minimum learning rate")
    parser.add_argument("--warm-up-iters", type=int, default=None, help="Number of warm-up iterations")
    parser.add_argument("--cosine-annealing-iters", type=int, default=None, help="Number of cosine annealing iterations")
    
    parser.add_argument("--use-gradient-clipping", type=bool, default=False, help="Whether to use gradient clipping")
    parser.add_argument("--max-norm-gradient-clipping", type=float, default=None, help="Maximum norm for gradient clipping")
    parser.add_argument("--eps-gradient-clipping", type=float, default=None, help="Epsilon value for gradient clipping, default to 1e-6")
    
    args = parser.parse_args()

    if args.iters <= 0:
        raise ValueError(f"Expected positive number of training iterations, got {args.iters}")

    if args.optim != "adamw":
        raise ValueError(f"Optimizer {args.optim} not implemented")
    
    if args.use_learning_rate_schedule:
        if (
            args.lr_max is None
            or args.lr_min is None
            or args.warm_up_iters is None
            or args.cosine_annealing_iters is None
        ):
            raise ValueError(
                "When --learning-rate-schedule is enabled, all of --lr-max, --lr-min, "
                "--warm-up-iters, and --cosine-annealing-iters must be specified"
            )
        
    if args.use_gradient_clipping:
        if args.max_norm_gradient_clipping is None or args.eps_gradient_clipping is None:
            raise ValueError("--max-norm-gradient-clipping must be specified when using gradient clipping")
    
    if args.checkpoint_save_path is not None and args.checkpoint_iters is None:
        raise ValueError("Argument --checkpoint-iters must be specified when --checkpoint-save-path is provided")

    main(args)
