# encoding: utf-8
"""
@author:  Huuy Q Can
@contact: huysk82000@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
from solver.build import make_lr_scheduler

import torch
import torch.nn.functional as F

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer

from utils.logger import setup_logger
from utils.checkpoint import Checkpointer
from utils.metric_logger import TensorboardLogger, MetricLogger
from utils.comm import synchronize, get_rank
from utils.directory import makedir


def train(cfg, local_rank, distributed, resume_from, use_tensorboard):
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )


    arguments = {}
    arguments["iteration"] = 0
    arguments["epoch"] = 0
    output_dir = cfg.OUTPUT_DIR
    
    save_to_disk = get_rank() == 0
    checkpointer = Checkpointer(
        model, optimizer, scheduler, output_dir, save_to_disk
    )
    
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    arguments["max_epoch"] = cfg.SOLVER.NUM_EPOCHS

    train_loader = make_data_loader(cfg, is_train=True)
    #val_loader = make_data_loader(cfg, is_train=False)
    
    if resume_from:
        if os.path.isfile(resume_from):
            extra_checkpoint_data = checkpointer.resume(resume_from)
            arguments.update(extra_checkpoint_data)
        else:
            raise IOError('{} is not a checkpoint file'.format(resume_from))
 

    
    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=os.path.join(output_dir, "tensorboard"),
            start_iter=arguments['iteration'],
            delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")

    do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
        checkpointer,
        meters,
        arguments
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch MOTION VAE Training")
    parser.add_argument(
        "--config-file", 
        default="", 
        help="path to config file",
        type=str
    )
    
    parser.add_argument(
        "--resume-from",
        default="",
        help='the checkpoint file to resume from',
        type=str
    )
    
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int
    )
    
    parser.add_argument(
        "opts", 
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    
    parser.add_argument(
        "--use-tensorboard",
        dest="use_tensorboard",
        help="Use tensorboardX logger (Requires tensorboardX and tensorflow installed)",
        action="store_true",
        default=False
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        makedir(output_dir)

    logger = setup_logger("MOTIONVAE", output_dir, get_rank())
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, args.resume_from, args.use_tensorboard)


if __name__ == '__main__':
    main()
