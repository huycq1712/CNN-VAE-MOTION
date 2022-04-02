# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .BaseVAE import BaseVAE
from .MOTIONVAE import MOTIONVAE


def build_model(cfg):
    in_channels = cfg.MODEL.MOTIONVAE.IN_CHANNELS
    latent_dim = cfg.MODEL.MOTIONVAE.LATENTDIM
    hidden_dims = cfg.MODEL.MOTIONVAE.HIDDEN_DIMS
    window_size = cfg.MODEL.MOTIONVAE.WINDOW_SIZE
    size_average = cfg.MODEL.MOTIONVAE.SIZE_AVG
    kld_weight = cfg.MODEL.MOTIONVAE.KLD_WEIGHT
    
    model = MOTIONVAE(in_channels=in_channels, 
                      latent_dim=latent_dim,
                      hidden_dims=hidden_dims,
                      kld_weight = kld_weight, 
                      window_size=window_size, 
                      size_average=size_average)
    return model
