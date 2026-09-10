"""Measure CUDA forward-only latency; exclude decoding files, transfer and post-processing."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
import ultralytics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--dhf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--warmup', type=int, default=50)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for precision in ('fp32', 'fp16'):
        for repeat in range(3):
            order = ('baseline', 'dhf') if repeat % 2 == 0 else ('dhf', 'baseline')
            for name in order:
                path = getattr(args, name)
                model = YOLO(str(path))
                model.fuse()
                net = model.model.eval().cuda()
                if precision == 'fp16':
                    net.half()
                x = torch.zeros((1, 3, 640, 640), device='cuda', dtype=next(net.parameters()).dtype)
                samples = []
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        net(x)
                    torch.cuda.synchronize()
                    for _ in range(args.iterations):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        net(x)
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end))
                row = dict(model=name, precision=precision, repeat=repeat,
                           checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           median_ms=float(np.median(samples)), mean_ms=float(np.mean(samples)),
                           p90_ms=float(np.percentile(samples, 90)), samples_ms=samples)
                results.append(row)
                print({k: v for k, v in row.items() if k != 'samples_ms'}, flush=True)
                del net, model, x
                torch.cuda.empty_cache()
    output = dict(created=time.time(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  ultralytics=ultralytics.__version__, batch=1, shape=[1, 3, 640, 640],
                  warmup_per_block=args.warmup, iterations_per_block=args.iterations,
                  cudnn_benchmark=False, tf32=False, results=results,
                  scope='Fused PyTorch eager forward on preallocated GPU tensor; not end-to-end latency.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
