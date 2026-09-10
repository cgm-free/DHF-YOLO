"""Run the predeclared submission controls sequentially, without changing training batch."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def jobs():
    return [
        dict(name='37_dhf_local_only_e100_s0', model='local_only', epochs=100, seed=0),
        dict(name='38_dhf_no_contrast_e100_s0', model='no_contrast', epochs=100, seed=0),
        dict(name='39_dhf_hdfm_batch16_e100_s0', model='dhf_hdfm', epochs=100, seed=0),
        dict(name='40_p2_e300_s1', model='p2', epochs=300, seed=1),
        dict(name='41_dhf_e300_s1', model='dhf', epochs=300, seed=1),
        dict(name='42_p2_e300_s2', model='p2', epochs=300, seed=2),
        dict(name='43_dhf_e300_s2', model='dhf', epochs=300, seed=2),
    ]


def prepare_configs(out, data_root):
    names = {'dhf': 'dhf-yolo26-p2.yaml', 'p2': 'yolo26-p2.yaml',
             'dhf_hdfm': 'dhfblock-hdfm-yolo26-p2.yaml'}
    configs = {key: yaml.safe_load((ROOT / 'ultralytics/cfg/models/26' / name).read_text())
               for key, name in names.items()}
    for kind in ('local_only', 'no_contrast'):
        cfg = yaml.safe_load(yaml.safe_dump(configs['dhf']))
        for layer in cfg['head']:
            if kind == 'local_only' and layer[2] == 'HDFMv2':
                layer[3].append(0.0)
            if kind == 'no_contrast' and layer[2] == 'C3k2DHF':
                layer[3] = [layer[3][0], True, 0.5, 1, True, 0.0]
        configs[kind] = cfg
    for key, cfg in configs.items():
        cfg['scale'] = 'n'
        cfg['scales'] = {'n': cfg['scales']['n']}
        cfg['nc'] = 10
        (out / f'yolo26n-{key}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    data = yaml.safe_load((ROOT / 'ultralytics/cfg/datasets/VisDrone2019-DET-repo.yaml').read_text())
    data['path'] = str(data_root.resolve())
    (out / 'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def train_job(args, job):
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    out = args.output.resolve()
    name = job['name'] + ('_smoke' if args.smoke else '')
    record = dict(job=job, state='starting', started=time.time(), source_hash=sha(__file__),
                  block_sha256=sha(ROOT / 'ultralytics/nn/modules/block.py'),
                  torch=torch.__version__, ultralytics=ultralytics.__version__,
                  gpu=torch.cuda.get_device_name(0), batch=16, initialization='random_from_yaml')
    state_path = out / 'status' / f'{name}.json'
    model_path = out / 'configs' / f"yolo26n-{job['model']}.yaml"
    record['model_yaml_sha256'] = sha(model_path)
    save_json(state_path, record)
    if (out / 'train' / name).exists():
        raise FileExistsError(f'Refusing to overwrite or silently resume {name}')

    def fixed_batch(trainer):
        trainer._oom_retries = 3
        if trainer.batch_size != 16:
            raise RuntimeError(f'Unmatched batch {trainer.batch_size}; expected 16')

    def progress(trainer):
        record.update(state='training', completed_epochs=trainer.epoch + 1,
                      updated=time.time(), batch=trainer.batch_size,
                      save_dir=str(trainer.save_dir))
        save_json(state_path, record)

    try:
        model = YOLO(str(model_path))
        params = get_num_params(model.model)
        if model.model.yaml.get('scale') != 'n' or not 2_000_000 < params < 3_000_000:
            raise RuntimeError(f'Unexpected model scale/size: {model.model.yaml.get("scale")}, {params}')
        record['unfused_parameters'] = params
        save_json(state_path, record)
        model.add_callback('on_train_epoch_start', fixed_batch)
        model.add_callback('on_fit_epoch_end', progress)
        model.train(data=str(out / 'configs/dataset.yaml'), imgsz=640, batch=16, device=0,
                    epochs=1 if args.smoke else job['epochs'], seed=job['seed'],
                    workers=8, box_loss='ciou', optimizer='MuSGD', lr0=0.01, momentum=0.9,
                    weight_decay=0.0005, lrf=0.01, cos_lr=False, pretrained=False,
                    deterministic=True, close_mosaic=0 if args.smoke else 10,
                    patience=0, amp=True, rect=False, fraction=0.01 if args.smoke else 1.0,
                    project=str(out / 'train'), name=name, exist_ok=False,
                    save_period=-1, plots=False)
        ckpt = Path(model.trainer.best)
        rows = list(csv.DictReader((ckpt.parents[1] / 'results.csv').open()))
        expected = 1 if args.smoke else job['epochs']
        if len(rows) != expected or model.trainer.batch_size != 16:
            raise RuntimeError('Incomplete epochs or unmatched batch; no publishable result')
        record.update(state='validating', checkpoint_sha256=sha(ckpt), checkpoint=str(ckpt))
        save_json(state_path, record)
        model = YOLO(str(ckpt))
        metrics = model.val(data=str(out / 'configs/dataset.yaml'), imgsz=640, batch=32,
                            device=0, workers=4, half=True, rect=True, split='val',
                            conf=0.001, iou=0.7, max_det=300, plots=True,
                            project=str(out / 'val'), name=name, exist_ok=False)
        raw = {key: float(value) for key, value in metrics.results_dict.items()}
        model.fuse()
        record.update(state='smoke_passed' if args.smoke else 'complete', completed=time.time(),
                      results=raw, parameters=get_num_params(model.model),
                      gflops=get_flops(model.model, imgsz=640), validation_directory=str(metrics.save_dir),
                      validation=dict(imgsz=640, batch=32, half=True, rect=True,
                                      split='val', conf=0.001, iou=0.7, max_det=300))
        save_json(state_path, record)
    except BaseException as exc:
        record.update(state='failed', updated=time.time(), error=f'{type(exc).__name__}: {exc}')
        save_json(state_path, record)
        raise


def run_queue(args):
    import fcntl
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = jobs()
    if args.job:
        selected = [j for j in selected if j['name'] == args.job]
        if not selected:
            raise ValueError(f'Unknown job {args.job}')
    manifest = dict(jobs=selected, batch=16, validation_batch=32, created=time.time(),
                    note='Pending experiments. No values are imputed or promoted to manuscript tables.',
                    source_root=str(ROOT), source_sha256=sha(__file__))
    save_json(out / ('smoke_manifest.json' if args.smoke else 'queue_manifest.json'), manifest)
    (out / 'logs').mkdir(exist_ok=True)
    for job in selected:
        name = job['name'] + ('_smoke' if args.smoke else '')
        state_path = out / 'status' / f'{name}.json'
        if state_path.exists():
            old = json.loads(state_path.read_text())
            if old['state'] in ('complete', 'smoke_passed'):
                continue
            raise RuntimeError(f'{name} already exists in state {old["state"]}; inspect before restarting')
        if shutil.disk_usage(out).free < 20 * 1024**3:
            raise RuntimeError('Less than 20 GiB free; queue stopped before next training')
        cmd = [sys.executable, str(Path(__file__).resolve()), '--output', str(out),
               '--worker', '--job', job['name']]
        if args.smoke:
            cmd.append('--smoke')
        save_json(out / 'queue_status.json', dict(state='running', job=name, updated=time.time()))
        with (out / 'logs' / f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if result.returncode:
            save_json(out / 'queue_status.json', dict(state='failed', job=name, updated=time.time(),
                                                    returncode=result.returncode))
            raise RuntimeError(f'{name} failed; remaining queue not started')
    save_json(out / 'queue_status.json', dict(state='complete', updated=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.prepare:
        if args.data_root is None:
            parser.error('--prepare requires --data-root')
        out = args.output.resolve() / 'configs'
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise FileExistsError('Prepared configs already exist; do not overwrite an active experiment')
        prepare_configs(out, args.data_root)
    elif args.worker:
        train_job(args, next(j for j in jobs() if j['name'] == args.job))
    else:
        run_queue(args)


if __name__ == '__main__':
    main()
