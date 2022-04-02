# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import time
import logging
import datetime

import torch
import torch.distributed as dist

from utils.comm import get_world_size, get_rank

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
        checkpointer,
        meters,
        arguments
):
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("MOTIONVAE.train")
    logger.info("Start training")
    max_iter = epochs * len(train_loader)
    iteration = arguments["iteration"] 
    distributed = arguments["distributed"]
    start_training_time = time.time()
    end = time.time()
    model.train()
    
    while epoch < epochs:
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        epoch = epoch + 1
        arguments["epoch"] = epoch
        scheduler.step()
        
        for step, (images,) in enumerate(train_loader):
            data_time = time.time() - end
            inner_iter = step
            iteration = iteration + 1
            arguments["iteration"] = iteration
            
            images = images.to(device)
            
            result = model(images)
            loss_dict = model.loss_function(result)
            losses = sum(loss for loss in loss_dict.values())
            
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in losses.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            batch_time = time.time() - end
            end = time.time()
            
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            if inner_iter % 1 == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch [{epoch}][{inner_iter}/{num_iter}]",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        inner_iter=inner_iter,
                        num_iter=len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[-1]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                
        if epoch % checkpoint_period == 0:
            checkpointer.save("epoch_{:d}".format(epoch), **arguments)
        if epoch == epochs:
            checkpointer.save("epoch_final", **arguments)
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

