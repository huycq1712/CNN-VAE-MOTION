# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch

def collate_fn(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0])
    
    return images