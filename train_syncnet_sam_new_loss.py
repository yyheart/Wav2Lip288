import math
import audio
import torch
import random
import datetime
import pandas as pd
import numpy as np
from torch import nn
from tqdm import tqdm
from time import time
from glob import glob
from torch import optim
from hparams import hparams
from dataloader_syncnet import Dataset
import os, random, cv2, argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from config.config import DINetTrainingOptions
# from models import SyncNet_color_384 as SyncNet
from models import SyncNetPerception
from dataloader_syncnet import DINetDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from os.path import dirname, join, basename, isfile


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=False,default="checkpoints/syncnet/",type=str)
parser.add_argument('--exp_num', help='ID number of the experiment', required=False, default="actor", type=str)
parser.add_argument('--history_train', help='Save history training', required=False,default="logs/syncnet/",type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint',required=False, default=None, type=str)
parser.add_argument("--lamb_syncnet_perception", help="weight of the sync loss", default=0.1, type=int)
args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
best_loss = 1000
print('use_cuda: {}'.format(use_cuda))
debug_list = []
opt = DINetTrainingOptions().parse_args()
train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)

criterionMSE = nn.MSELoss().cuda()
real_tensor = torch.tensor(1.0).cuda()
logloss = nn.BCELoss().cuda()
def cosine_loss(a, v):
    d = - torch.log((1 + nn.functional.cosine_similarity(a, v))/2)
    loss = logloss(d,  real_tensor.expand_as(d))
    return loss



def train(device, model, train_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    logger = CSVLogger(args.history_train, name=args.exp_num)

    stop_training = False
    while global_epoch < nepochs:
        st_e = time()
        try:
            print('Starting Epoch: {}'.format(global_epoch))
            running_loss = 0.
            for step, data in tqdm(enumerate(train_data_loader)):
                st = time()
                model.train()
                (
                    source_clip,
                    deep_speech_full
                ) = data
                # print("source_clip", source_clip.shape) # [24, 5, 3, 104, 80]
                # to cuda
                deep_speech_full = deep_speech_full.float().cuda()
                source_clip = (
                    torch.cat(torch.split(source_clip, 1, dim=1), 0)
                    .squeeze(1)
                    .float()
                    .cuda()
                )
                # print("source_clip", source_clip.shape)  # torch.Size([120, 3, 104, 80])
                fake_out_clip = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
                fake_out_clip_mouth = fake_out_clip[
                    :, # B
                    :, # C
                    train_data.radius : train_data.radius + train_data.mouth_region_size,
                    train_data.radius_1_4 : train_data.radius_1_4
                    + train_data.mouth_region_size,
                ]
                # print("fake_out_clip_mouth", fake_out_clip_mouth.shape) # torch.Size([24, 15, 256, 256])
                # mouth region    
                # hubert_feature  replace 
                face_embedding, audio_embedding = model(fake_out_clip_mouth, deep_speech_full) # need be 0~1
                # sync_score = sync_score.cuda() 
                face_embedding.cuda()
                audio_embedding.cuda()

                # max_value = torch.max(sync_score)
                # min_value = torch.min(sync_score)
                # should be  ([B, 1, 8, 8])
                loss_sync = cosine_loss(face_embedding, audio_embedding)
                loss_sync.backward()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                running_loss += loss_sync.item()
                
                print(f"Step {global_step} | Loss: {running_loss/(step+1):.8f} | Elapsed: {(time() - st):.5f}")
                if global_step % 500 == 0:
                    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
                    logger.log_metrics({
                        "train_loss": running_loss / (step + 1),
                        },step=global_step)
                    logger.save()
                    model.train()
            if stop_training:
                print("The model has converged, stop training.")
                break
            print("Epoch time:", time() - st_e)
            global_epoch += 1
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break
    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, 1000)
    logger.save()


def save_ckpt(model, optimizer, step, checkpoint_dir, epoch, model_name):
    checkpoint_path = join(checkpoint_dir, model_name)
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "best_loss": best_loss,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)
#  remove the evaluate part

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    # save best.pth
    global best_loss
    date = str(datetime.datetime.now()).split(" ")[0]
    post_fix = f'checkpoint_{hparams.img_size}_{hparams.syncnet_batch_size}_{global_step:09d}_{date}.pth'
    save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"last_syncnet_{args.exp_num}.pth")

    prefix = "syncnet_"
    save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"{prefix}{post_fix}")

    ckpt_list = os.listdir(checkpoint_dir)
    ckpt_list = [file for file in ckpt_list if prefix in file and "checkpoint_" in file and "syncnet_" in file]
    num_ckpts = hparams.num_checkpoints
    if len(ckpt_list) <= num_ckpts*2:
        return

    ckpt_list.sort(key=lambda x: int(x.replace(".pth", "").split("_")[-2]))
    num_elim = len(ckpt_list) - num_ckpts
    elim_ckpt = ckpt_list[:num_elim]
    for ckpt in elim_ckpt:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        os.remove(ckpt_path)
        print("Deleted", ckpt_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch
    global best_loss

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    best_loss = checkpoint["best_loss"]
    return model


def run():
    # global global_step
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_num)
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    # 换成DINet
    train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)
    training_data_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=12,
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SyncNetPerception().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    else:
        print("Training From Scratch !!!")
    train(device, model, training_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)

if __name__ == "__main__":
    run()