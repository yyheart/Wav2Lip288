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

from models import SyncNet_color_384 as SyncNet
# from models import SyncNetPerception


from pytorch_lightning.loggers import CSVLogger
from os.path import dirname, join, basename, isfile


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=False,default="checkpoints/syncnet/",type=str)
parser.add_argument('--exp_num', help='ID number of the experiment', required=False, default="actor", type=str)
parser.add_argument('--history_train', help='Save history training', required=False,default="logs/syncnet/",type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default="/mnt/sdb/liwen/wav2lip_288x288/checkpoints/syncnet/actor/syncnet_checkpoint_384_128_000002500_2023-12-31.pth", type=str)
args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
best_loss = 1000
print('use_cuda: {}'.format(use_cuda))

logloss = nn.BCELoss()  # 交叉熵
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
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
            for step, (x, mel, y) in tqdm(enumerate(train_data_loader)):

                st = time()
                model.train()
                optimizer.zero_grad()
                x = x.to(device)

                mel = mel.to(device)
                y = y.to(device)
                a, v = model(mel, x)
                # print(x.shape)  # [64, 15, 192, 384]
                loss = cosine_loss(a, v, y)

                loss.backward()
                optimizer.step()
                d = nn.functional.cosine_similarity(a, v)
                global_step += 1

                cur_session_steps = global_step - resumed_step
                running_loss += loss.item()

                print(f"Step {global_step} | out_of_sync_distance: {d.detach().cpu().clone().numpy().mean():.8f} | Loss: {running_loss/(step+1):.8f} | Elapsed: {(time() - st):.5f}")
                # if global_step == 1 or global_step % checkpoint_interval == 0:

                if global_step % 500 == 0:
                    with torch.no_grad():
                        model.eval()
                        eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                        # if eval_loss < 0.25:
                        #     stop_training = True
                    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, eval_loss)
                    logger.log_metrics({
                        "train_loss": running_loss / (step + 1),
                        "eval_loss": eval_loss
                        },step=global_step)
                    logger.save()
                    model.train()
                del x, mel, y
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


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 10
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    for step, (x, mel, y) in enumerate(test_data_loader):
        print("Eval step", step)
        # model.eval()
        # Transform data to CUDA device
        x = x.to(device)
        mel = mel.to(device)
        a, v = model(mel, x)
        y = y.to(device)
        loss = cosine_loss(a, v, y)
        print("Eval loss", loss.item())
        losses.append(loss.item())

    averaged_loss = sum(losses) / len(losses)
    print(averaged_loss)
    return averaged_loss


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


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, loss_val):
    # save best.pth
    global best_loss
    date = str(datetime.datetime.now()).split(" ")[0]
    post_fix = f'checkpoint_{hparams.img_size}_{hparams.syncnet_batch_size}_{global_step:09d}_{date}.pth'
    if loss_val < best_loss:
        best_loss = loss_val
        save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"best_syncnet_{args.exp_num}.pth")
    # last model
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

    train_dataset = Dataset('train_lrs2_remove')
    test_dataset = Dataset('val_lrs2_remove')
    # HParams.set_hparam("syncnet_batch_size", 4)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers,
        drop_last=True
    )

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=hparams.num_workers,
        drop_last=True
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    # model = nn.DataParallel(SyncNet()).to(device) # 模型
    model = SyncNet().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    else:
        print("Training From Scratch !!!")

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)


if __name__ == "__main__":
    run()