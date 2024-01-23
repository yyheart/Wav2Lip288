import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 将data写成一个json文件输入到网络中
def get_data(json_name, augment_num):
    print("start loading data")
    with open(json_name, "r") as f: 
        data_dic = json.load(f) # 加载一个字典
    data_dic_name_list = []
    for augment_index in range(augment_num): # 加多少个参数
        for video_name in data_dic.keys():
            data_dic_name_list.append(video_name)  # keys是是视频的名字
    random.shuffle(data_dic_name_list)
    print("finish loading")
    return data_dic_name_list, data_dic


# 如果嘴部区域是64那么取的图像大小为112*80 ,其他的大小同理
class DINetDataset(Dataset):
    def __init__(self, path_json, augment_num, mouth_region_size):
        super(DINetDataset, self).__init__()
        self.data_dic_name_list, self.data_dic = get_data(path_json, augment_num)
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size // 2  # 嘴部区  64/2
        self.radius_1_4 = self.radius // 4    # 嘴部区  1/4的半径
        self.img_h = self.radius * 3 + self.radius_1_4  # 3*64/2 + 64/4=112
        self.img_w = self.radius * 2 + self.radius_1_4 * 2 # 2*64/2 + 64/4 = 80
        self.length = len(self.data_dic_name_list)

    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]["clip_data_list"])

        source_anchor = random.sample(range(video_clip_num-1), 1)[0]
        
        source_image_path_list = self.data_dic[video_name]["clip_data_list"][source_anchor]["frame_path_list"]
        source_clip_list = []
        source_clip_mask_list = []
        deep_speech_list = []
        reference_clip_list = []
        for source_frame_index in range(2, 2 + 5):
            ## load source clip
            source_image_data = cv2.imread(source_image_path_list[source_frame_index])[
                :, :, ::-1
            ]
            source_image_data = (
                cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
            )
            source_clip_list.append(source_image_data)
            source_image_mask = source_image_data.copy()
            source_image_mask[
                self.radius : self.radius + self.mouth_region_size,
                self.radius_1_4 : self.radius_1_4 + self.mouth_region_size,
                :,
            ] = 0
            source_clip_mask_list.append(source_image_mask)

            ## load deep speech feature
            deepspeech_array = np.array(
                self.data_dic[video_name]["clip_data_list"][source_anchor][
                    "deep_speech_list"
                ][source_frame_index - 2 : source_frame_index + 3]
            )
            deep_speech_list.append(deepspeech_array)

            ## ## load reference images
            reference_frame_list = []
            reference_anchor_list = random.sample(range(video_clip_num), 5)
            for reference_anchor in reference_anchor_list:
                reference_frame_path_list = self.data_dic[video_name]["clip_data_list"][
                    reference_anchor
                ]["frame_path_list"]
                reference_random_index = random.sample(range(9), 1)[0]
                reference_frame_path = reference_frame_path_list[reference_random_index]
                reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]
                reference_frame_data = (
                    cv2.resize(reference_frame_data, (self.img_w, self.img_h)) / 255.0
                )
                reference_frame_list.append(reference_frame_data)
            reference_clip_list.append(np.concatenate(reference_frame_list, 2))

        source_clip = np.stack(source_clip_list, 0)
        source_clip_mask = np.stack(source_clip_mask_list, 0)
        deep_speech_clip = np.stack(deep_speech_list, 0)
        reference_clip = np.stack(reference_clip_list, 0)
        deep_speech_full = np.array(
            self.data_dic[video_name]["clip_data_list"][source_anchor][
                "deep_speech_list"
            ]
        )

        source_clip = torch.from_numpy(source_clip).float().permute(0, 3, 1, 2)
        source_clip_mask = (
            torch.from_numpy(source_clip_mask).float().permute(0, 3, 1, 2)
        )
        reference_clip = torch.from_numpy(reference_clip).float().permute(0, 3, 1, 2)
        deep_speech_clip = torch.from_numpy(deep_speech_clip).float().permute(0, 2, 1)
        deep_speech_full = torch.from_numpy(deep_speech_full).permute(1, 0)
        return (
            source_clip_mask,
            deep_speech_full
        )

    def __len__(self):
        return self.length

if __name__=="__main__":
    from config.config import DINetTrainingOptions
    opt = DINetTrainingOptions().parse_args()
    train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)
    training_data_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    
    