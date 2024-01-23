import math
import audio
import torch
import numpy as np
import pandas as pd
from torch import nn
from time import time
from glob import glob
from torch import optim
import os, random, cv2, argparse
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from hparams import hparams, get_image_list_new
from os.path import dirname, join, basename, isfile

use_cuda = torch.cuda.is_available()
syncnet_T = 5
syncnet_mel_step_size = 16
# mel augmentation
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
def get_audio_length(audio_path):
    """Get the length of the audio file in seconds"""
    cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(audio_path)
    audio_length = float(os.popen(cmd).read().strip())
    return audio_length


class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list_new(split)   # 路径 /5566854121/01

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame) 
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            # frame = join(vidname, f'{frame_id:05}.jpg')
            frame =join(vidname, f'{frame_id}.jpg')
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx : end_idx, :]
    
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T: # 小于48张图不要
                continue

            # img_name = random.choice(img_names)
            # wrong_img_name = random.choice(img_names)

            img_name_idx = random.randint(0, len(img_names) - syncnet_T + 1)
            wrong_img_name_idx = random.randint(0, len(img_names) - syncnet_T + 1)
            
            # 表示的是取第...个元素
            img_name = img_names[img_name_idx - 1]
            wrong_img_name = img_names[wrong_img_name_idx - 1]

            count_same = 0
            while wrong_img_name == img_name:  # 选到相同的了
                wrong_img_name = random.choice(img_names)
                count_same += 1
                if count_same > 10:  # 连续选到10个相同的就是有问题了，说明这个文件夹可能只有这一个图片
                    break
            if count_same > 10:  # 结束本次循环
                continue
            
            # 随机选择 y=1 或者y=0
            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name
            
            # /data2/cxh/liwen/work/Wav2Lip/lrs2_preprocessed/6092631012156491804/00002/54.jpg
            # print(chosen)   
            window_fnames = self.get_window(chosen)

            # print(window_fnames) # None
            # exit(0)
            if window_fnames is None:
                # print("window_fnames")
                continue 

            window = []
            all_read = True
            is_flip = random.random() < 0.5
            for fname in window_fnames:
                try:
                    img = cv2.imread(fname)
                    if is_flip:
                        img = cv2.flip(img, 1)
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    print(e)
                    all_read = False
                    break
                window.append(img)

            if not all_read:
                print("if not all_read:")
                continue
            try:
                mel_out_path = join(vidname, "mel.npy")
                if os.path.isfile(mel_out_path):  # x50 times faster - 0.002 -> 0.01s
                    with open(mel_out_path, "rb") as f:
                        orig_mel = np.load(f)
                else:
                    # print("this way")
                    wavpath = os.path.join(vidname, "audio.wav")
                    wav = audio.load_wav(wavpath, hparams.sample_rate)

                    orig_mel = audio.melspectrogram(wav).T  # 0.2 -> 0.9s
                    with open(mel_out_path, "wb") as f:
                        np.save(f, orig_mel)
            except Exception as e:
                # print("mel", vidname)
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            # mel augmentation
            if random.random() < 0.3:
                mel = mask_mel(mel)

            del orig_mel

            if (mel.shape[0] != syncnet_mel_step_size):
                # print("Mel shape")
                continue

            # H x W x 3 * T
            # x = np.concatenate(window, axis=2) / 255. # [0, 1]
            x = (np.concatenate(window, axis=2) / 255.0)
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            return x, mel, y

if __name__ =="__main__":
    data_root = "/mnt/sdb/liwen/wav2lip_288x288/avspeech_frame"
    train_dataset = Dataset('train_lrs2_remove')
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers,
        drop_last=True)
    data_iter = iter(train_data_loader)
    x, mel, y = next(data_iter)
    print(x.shape)