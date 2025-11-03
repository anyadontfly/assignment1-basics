import argparse
import logging
import sys
import timeit

import torch
import torch.cuda.nvtx as nvtx

import cse599o_basics.model
from cse599o_basics.model import Transformer, annotated_scaled_dot_product_attention
from cse599o_basics.utils import AdamW, cross_entropy_loss


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)

logger.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer model")

    parser.add_argument("--batch-size", type=int, required=True, help="Training batch size")    
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"], help="Model data type. One of: float32, bfloat16, or float16 (default: float32)")
    parser.add_argument("--dim", type=int, required=True, help="Model hidden dimension size")
    parser.add_argument("--n-layers", type=int, required=True, help="Number of Transformer blocks")
    parser.add_argument("--n-heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--rope-theta", type=int, required=True, help="Base frequency for rotary positional embeddings")
    parser.add_argument("--ffn-dim", type=int, required=True, help="Model feed-forward layer dimension size")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size of dataset")
    parser.add_argument("--context-length", type=int, required=True, help="Maximum sequence length")

    parser.add_argument("--optim", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer name")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay, default to 0.0")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 (first momentum) for optimizer, default to 0.9")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 (second momentum) for optimizer, default to 0.999")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon value for optimizer stability, default to 1e-8")

    parser.add_argument("--iters-warmup", type=int, required=True, help="Total number of training iterations")
    parser.add_argument("--iters-benchmark", type=int, required=True, help="Total number of training iterations")
    parser.add_argument("--profile-range", type=str, default="fb", choices=["f", "fb", "fbo"], help="Ranges to profile: f (forward), b (backward), o (optimizer step)")
    parser.add_argument("--mem", action="store_true", help="Enable memory profiling")

    args = parser.parse_args()

    if args.iters_warmup <= 0:
        raise ValueError(f"Expected positive number of warmup training iterations, got {args.iters_warmup}")
    if args.iters_benchmark <= 0:
        raise ValueError(f"Expected positive number of benchmark training iterations, got {args.iters_benchmark}")
    if args.optim != "adamw":
        raise ValueError(f"Optimizer {args.optim} not implemented")

    return args


def setup_model_and_optimizer(args, device, dtype):
    model = Transformer(
        args.dim,
        args.n_heads,
        args.ffn_dim,
        args.vocab_size,
        args.context_length,
        args.n_layers,
        args.rope_theta,
        dtype=dtype,
    ).to(device)
    
    if args.profile_range == "f":
        return model, None
    
    optim = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    
    return model, optim


def warmup(model, optim, inputs, targets, vocab_size, iters, profile_range):
    logger.info(f"Starting {iters} warmup iterations")
    
    for _ in range(iters):
        if profile_range == "f":
            with torch.no_grad():
                logits = model(inputs)
            continue
        
        logits = model(inputs)
        loss = cross_entropy_loss(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        
        if optim is not None:
            optim.zero_grad()
        
        if profile_range == "fb":
            continue
        
        if optim is not None:
            optim.step()
    
    torch.cuda.synchronize()


def run_benchmark(model, optim, inputs, targets, vocab_size, iters, profile_range, n_layers):
    logger.info(f"Starting {iters} benchmark iterations")
    
    forward_events = []
    backward_events = []
    atten_events = []
    ffn_events = []
    forward_cpu_times = []
    backward_cpu_times = []
    iter_cpu_times = []
    
    for _ in range(iters):
        forward_start_event = torch.cuda.Event(enable_timing=True)
        forward_end_event = torch.cuda.Event(enable_timing=True)
        backward_start_event = torch.cuda.Event(enable_timing=True)
        backward_end_event = torch.cuda.Event(enable_timing=True)

        iter_cpu_time_start = timeit.default_timer()
        
        forward_start_event.record()
        forward_cpu_time_start = timeit.default_timer()
        
        if profile_range == "f":
            with torch.no_grad():
                with nvtx.range("forward pass"):
                    logits = model(inputs, times=(atten_events, ffn_events))
        else:
            with nvtx.range("forward pass"):
                logits = model(inputs, times=(atten_events, ffn_events))
        
        forward_end_event.record()
        forward_cpu_time_end = timeit.default_timer()
        
        forward_events.extend([forward_start_event, forward_end_event])
        forward_cpu_times.append((forward_cpu_time_end - forward_cpu_time_start) * 1000)
        
        if profile_range == "f":
            iter_cpu_time_end = timeit.default_timer()
            iter_cpu_times.append((iter_cpu_time_end - iter_cpu_time_start) * 1000)
            continue
        
        loss = cross_entropy_loss(logits.view(-1, vocab_size), targets.view(-1))

        backward_start_event.record()
        backward_cpu_time_start = timeit.default_timer()
        
        loss.backward()
        
        backward_end_event.record()
        backward_cpu_time_end = timeit.default_timer()
        
        backward_events.extend([backward_start_event, backward_end_event])
        backward_cpu_times.append((backward_cpu_time_end - backward_cpu_time_start) * 1000)
        
        if optim is not None:
            with nvtx.range("zero grad"):
                optim.zero_grad()

        if profile_range == "fb":
            iter_cpu_time_end = timeit.default_timer()
            iter_cpu_times.append((iter_cpu_time_end - iter_cpu_time_start) * 1000)
            continue
        
        if optim is not None:
            with nvtx.range("optimizer step"):
                optim.step()

        iter_cpu_time_end = timeit.default_timer()
        iter_cpu_times.append((iter_cpu_time_end - iter_cpu_time_start) * 1000)

    torch.cuda.synchronize()

    forward_times = [forward_events[2*i].elapsed_time(forward_events[2*i+1]) for i in range(iters)]
    backward_times = [backward_events[2*i].elapsed_time(backward_events[2*i+1]) for i in range(iters)] if profile_range != "f" else []
    
    atten_times = []
    ffn_times = []
    for i in range(iters):
        for j in range(n_layers):
            atten_event_idx = (i * n_layers + j) * 2
            ffn_event_idx = (i * n_layers + j) * 2
            atten_times.append(atten_events[atten_event_idx].elapsed_time(atten_events[atten_event_idx + 1]))
            ffn_times.append(ffn_events[ffn_event_idx].elapsed_time(ffn_events[ffn_event_idx + 1]))

    return {
        'forward_times': forward_times,
        'backward_times': backward_times,
        'atten_times': atten_times,
        'ffn_times': ffn_times,
        'forward_cpu_times': forward_cpu_times,
        'backward_cpu_times': backward_cpu_times,
        'iter_cpu_times': iter_cpu_times,
    }


def report_results(results, profile_range, iters_benchmark):
    forward_times_tensor = torch.tensor(results['forward_times'])
    backward_times_tensor = torch.tensor(results['backward_times']) if results['backward_times'] else torch.tensor([0.0])
    atten_times_tensor = torch.tensor(results['atten_times'])
    ffn_times_tensor = torch.tensor(results['ffn_times'])
    
    logger.info("=" * 60)
    logger.info(f"Profile range: {profile_range}")
    logger.info("=" * 60)
    logger.info("GPU Timing (CUDA Events):")
    logger.info(f"  Forward:  {forward_times_tensor.mean():.3f} ms ± {forward_times_tensor.std():.3f} ms")
    if profile_range != "f":
        logger.info(f"  Backward: {backward_times_tensor.mean():.3f} ms ± {backward_times_tensor.std():.3f} ms")
    logger.info("")
    logger.info("GPU Timing per Layer (CUDA Events):")
    logger.info(f"  Attention: {atten_times_tensor.mean():.3f} ms ± {atten_times_tensor.std():.3f} ms")
    logger.info(f"  FFN:       {ffn_times_tensor.mean():.3f} ms ± {ffn_times_tensor.std():.3f} ms")
    logger.info("")
    logger.info("CPU Timing (Wall Clock):")
    logger.info(f"  Iteration: {sum(results['iter_cpu_times']) / iters_benchmark:.3f} ms")
    logger.info(f"  Forward:   {sum(results['forward_cpu_times']) / iters_benchmark:.3f} ms")
    if profile_range != "f":
        logger.info(f"  Backward:  {sum(results['backward_cpu_times']) / iters_benchmark:.3f} ms")
    logger.info("=" * 60)


def main(args):
    cse599o_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]
    torch.set_default_dtype(dtype)
    
    logger.info(f"Initializing model on device {device} with dtype {torch.get_default_dtype()}")
    
    model, optim = setup_model_and_optimizer(args, device, dtype)
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    
    model.train()
    
    warmup(model, optim, inputs, targets, args.vocab_size, args.iters_warmup, args.profile_range)
    
    torch.cuda.empty_cache()
    
    if args.mem:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    results = run_benchmark(model, optim, inputs, targets, args.vocab_size, args.iters_benchmark, args.profile_range, args.n_layers)
    
    if args.mem:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
    report_results(results, args.profile_range, args.iters_benchmark)


if __name__ == "__main__":
    args = parse_args()
    main(args)
