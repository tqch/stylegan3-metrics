# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import click
import json
import tempfile
import torch
import torch.distributed
import torch.backends.cuda
import torch.backends.cudnn

import metrics.dnnlib as dnnlib
from metrics.metrics import metric_main
from metrics.metrics import metric_utils
from metrics.torch_utils import training_stats
from metrics.torch_utils import custom_ops
from metrics.torch_utils.ops import conv2d_gradfix
from datasets import DATASET_DICT, to_numpy3

# ----------------------------------------------------------------------------


def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank,
                                                 world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank,
                                                 world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Configure torch.
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(
            metric=metric, dataset_kwargs=args.dataset_kwargs, samp_kwargs=args.samp_kwargs,
            num_gpus=args.num_gpus, rank=rank, device=device, progress=progress)
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, sample_folder=args.sample_folder)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')


# ----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


# ----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list,
              default='fid50k_full', show_default=True)
@click.option('--data-root', help='Root directory of datasets', metavar='STR')
@click.option('--dataset', help='Pre-registered dataset to evaluate against', metavar='STR')
@click.option('--data-folder', help='Folder of reference data', metavar='DIR')
@click.option('--sample-folder', help='Folder of generated samples', metavar='DIR', required=True)
@click.option('--mirror', help='Enable dataset x-flips  [default: look up]', type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL',
              show_default=True)
def calc_metrics(ctx, metrics, data_root, dataset, data_folder, sample_folder, mirror, gpus, verbose):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=eqt50k_int,eqr50k \\
        --network=~/training-runs/00000-stylegan3-r-mydataset/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq-1024x1024.zip --mirror=1 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

    \b
    Recommended metrics:
      fid50k_full  Frechet inception distance against the full dataset.
      kid50k_full  Kernel inception distance against the full dataset.
      pr50k3_full  Precision and recall againt the full dataset.
      ppl2_wend    Perceptual path length in W, endpoints, full image.
      eqt50k_int   Equivariance w.r.t. integer translation (EQ-T).
      eqt50k_frac  Equivariance w.r.t. fractional translation (EQ-T_frac).
      eqr50k       Equivariance w.r.t. rotation (EQ-R).

    \b
    Legacy metrics:
      fid50k       Frechet inception distance against 50k real images.
      kid50k       Kernel inception distance against 50k real images.
      pr50k3       Precision and recall against 50k real images.
      is50k        Inception score for CIFAR-10.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, sample_folder=sample_folder, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Initialize dataset options.
    if dataset is not None:
        class_name = ".".join(['datasets', DATASET_DICT[dataset].__name__])
        dataset_kwargs = dict()
        if data_root is not None:
            dataset_kwargs['root'] = data_root
        dataset_kwargs['hflip'] = mirror
        dataset_kwargs['out_type'] = 'numpy'
        args.dataset_kwargs = dnnlib.EasyDict(class_name=class_name, **dataset_kwargs)
    elif data_folder is not None:
        if os.path.isdir(data_folder):
            args.dataset_kwargs = dnnlib.EasyDict(
                class_name='datasets.ImageFolder', root=data_folder, transform=to_numpy3)
        elif os.path.isfile(data_folder) and data_folder.endswith(".npz"):
            args.dataset_kwargs = dnnlib.EasyDict(
                class_name='datasets.NPZLoader', npz_file=data_folder, transform=to_numpy3)
        else:
            raise NotImplementedError("Unsupported data folder format!")
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Generated dataset
    assert os.path.exists(sample_folder), "sample_folder provided does not exists!"
    if os.path.isdir(sample_folder):
        args.samp_kwargs = dnnlib.EasyDict(class_name='datasets.ImageFolder', root=sample_folder, transform=to_numpy3)
    elif os.path.isfile(sample_folder) and sample_folder.endswith(".npz"):
        args.samp_kwargs = dnnlib.EasyDict(class_name='datasets.NPZLoader', npz_file=sample_folder, transform=to_numpy3)
    else:
        raise NotImplementedError("Unsupported sample folder format!")

    args.run_dir = os.path.dirname(sample_folder.rstrip(r"\/"))

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
