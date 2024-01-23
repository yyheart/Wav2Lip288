from os.path import dirname, join, basename, isfile
import math
import sys
from tqdm import tqdm
from random import shuffle
import pandas as pd
import time
import datetime
from lpips import LPIPS
import audio
import torchvision.transforms as T
import torch
import logging
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from random import shuffle
from glob import glob
import os, random, cv2, argparse
from hparams import hparams, get_image_list_new
import torch.multiprocessing as mp
import torch.distributed as dist
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


def mask_mel(crop_mel):
    block_size = 0.1
    time_size = math.ceil(block_size * crop_mel.shape[0])
    freq_size = math.ceil(block_size * crop_mel.shape[1])
    time_lim = crop_mel.shape[0] - time_size
    freq_lim = crop_mel.shape[1] - freq_size

    time_st = random.randint(0, time_lim)
    freq_st = random.randint(0, freq_lim)

    mel = crop_mel.copy()
    mel[time_st:time_st+time_size] = -4.
    mel[:, freq_st:freq_st + freq_size] = -4.

    return mel

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list_new(split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, f'{frame_id}.jpg')
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames
    
    def read_wrong_window(self, window_fnames, is_flip):
        if window_fnames is None: return None, 0, 0, 0
        shuffle(window_fnames)
        if random.random() > 0.5:
            window_fnames = [random.choice(window_fnames)]*len(window_fnames)
        window = []
        h, w, c = 0, 0, 0
        for fname in window_fnames:
            try:
                img = cv2.imread(fname)
                h, w, c = img.shape
                if is_flip:
                    img = cv2.flip(img, 1)
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None
            window.append(img)
        return window, h, w, c
    
    def read_window(self, window_fnames, is_flip):
        if window_fnames is None: 
            # print("77777777")
            return None, 0, 0, 0
        window = []
        h, w, c = 0, 0, 0
        for fname in window_fnames:
            try:
                img = cv2.imread(fname)
                h, w, c = img.shape
                if is_flip:
                    img = cv2.flip(img, 1)
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None
            window.append(img)
        # print("2222")
        return window, h, w, c

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        # b x h x w x c -> tensor c x b x h x w
        x = (np.asarray(window))/255.0
        x = np.transpose(x, (3, 0, 1, 2))

        return x
    
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            is_silence = random.random() > 0.5
            is_flip = random.random() > 0.7  # 随机选择翻转

            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                # print("Len", vidname)
                continue

            img_name = random.choice(img_names)
            # print(img_name)

            id_img_name = self.get_frame_id(img_name)
            wrong_img_name = img_names[(id_img_name + 5) % len(img_names)]
            id_wrong_img_name = self.get_frame_id(wrong_img_name)
            while wrong_img_name == img_name or abs(id_img_name - id_wrong_img_name) < 5:
                
                wrong_img_name = random.choice(img_names)
                id_wrong_img_name = self.get_frame_id(wrong_img_name)
                
            
            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            

            # print("1111")
            window, h, w, c  = self.read_window(window_fnames, is_flip)
            if window is None:
                continue
            
            wrong_window, h, w, c = self.read_wrong_window(wrong_window_fnames, is_flip)
            if wrong_window is None:
                continue
            
            # print("1111111111111")
            try:
                mel_out_path = join(vidname, "mel.npy")
               
                if not mel_out_path.endswith(".wav") and os.path.isfile(mel_out_path):  # x50 times faster - 0.002 -> 0.01s
                    with open(mel_out_path, "rb") as f:
                        orig_mel = np.load(f)
                else:
                    wavpath = os.path.join(vidname, "synced.wav")
                    
                    if not os.path.isfile(wavpath):
                        au_names = list(glob(join(vidname, '*.wav')))
                        au_path = au_names[0]
                        status = os.system(f"ffmpeg -i {au_path} -ar 16000 {wavpath}")

                    wav = audio.load_wav(wavpath, hparams.sample_rate)
                    orig_mel = audio.melspectrogram(wav).T  # 0.2 -> 0.9s
                    with open(mel_out_path, "wb") as f:
                        np.save(f, orig_mel)
            except Exception as e:
                continue
            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                # print("Mel shape", vidname)
                continue
                
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None:
                continue
            
            # print("111")
            # ground truth images
            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2 :,:] = 0

            # reference images
            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y, vidname


if __name__ =="__main__":
    train_dataset = Dataset('av_speech_select_remove')
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers,
        drop_last=True
)
    data_iter = iter(train_data_loader)
    x, indiv_mels, mel, y, vidname = next(data_iter)
    print(x.shape)