import torch
from torch import nn
from models.aps_models.apspool import ApsPool
from models.circular_pad_layer import circular_pad

class ResBlock1d(nn.Module):
    """
    basic block (no BN)
    """
    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features, out_features, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out


class ResBlock2d(nn.Module):
    """
    basic block (no BN)
    """
    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features, out_features, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

# class DownBlock1d(nn.Module):
#     """
#     basic block (no BN)
#     """

#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(DownBlock1d, self).__init__()
#         self.conv = nn.Conv1d(
#             in_channels=in_features,
#             out_channels=out_features,
#             kernel_size=kernel_size,
#             padding=padding,
#             stride=2,
#         )
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.relu(out)
#         return out

class DownBlock1d(nn.Module):
    """
    basic block (no BN)
    """

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(DownBlock1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

# After conv add downsample
class DownBlock2d(nn.Module):
    """
    basic block (no BN)
    """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, circular_pad = False, apspool_criterion = 'l2', N=None, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(  # 在没有显示的指定卷积的情况下，stride=1
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.circular_flag = circular_pad
        self.apspool_criterion = apspool_criterion
        self.N = N
        self.stride = stride
        # self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.pool = ApsPool(out_features, 
                            filt_size = 2,   
                            stride = self.stride, 
                            circular_flag = self.circular_flag, 
                            apspool_criterion = self.apspool_criterion, 
                            return_poly_indices = True, 
                            N = self.N)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        if self.stride>1:
            out, _ = self.pool(out)
        return out

# 1D卷积加个激活函数
class SameBlock1d(nn.Module):
    """
    basic block (no BN)
    """
    def __init__(self, in_features, out_features, kernel_size, padding):
        super(SameBlock1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

# 2D卷积加激活函数
class SameBlock2d(nn.Module):
    """
    basic block (no BN)
    """
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class FaceEncoder(nn.Module):
    """
    image encoder
    """
    def __init__(self, in_channel, out_dim):
        super(FaceEncoder, self).__init__()
        self.in_channel = in_channel
        self.out_dim = out_dim
        self.face_conv = nn.Sequential(
            SameBlock2d(in_channel, 64, kernel_size=7, padding=3),  # 29 64 7 3
            # # 64 → 32
            ResBlock2d(64, 64, kernel_size=3, padding=1),   # 64 64 3 1
            DownBlock2d(64, 64, 3, 1),  # 64 64 
            SameBlock2d(64, 128),
            # 32 → 16
            ResBlock2d(128, 128, kernel_size=3, padding=1), # 
            DownBlock2d(128, 128, 3, 1),
            SameBlock2d(128, 128),
            # 16 → 8
            ResBlock2d(128, 128, kernel_size=3, padding=1), # 
            DownBlock2d(128, 128, 3, 1),
            SameBlock2d(128, 128),
            # 8 → 4
            ResBlock2d(128, 128, kernel_size=3, padding=1), # 
            DownBlock2d(128, 128, 3, 1),
            SameBlock2d(128, 128),
            # 4 → 2
            ResBlock2d(128, 128, kernel_size=3, padding=1), # 
            DownBlock2d(128, 128, 3, 1),
            SameBlock2d(128, out_dim, kernel_size=1, padding=0),
        )

    def forward(self, x):
        ## b x c x h x w
        out = self.face_conv(x)
        return out


class AudioEncoder(nn.Module):
    """
    audio encoder
    """
    def __init__(self, in_channel, out_dim):
        super(AudioEncoder, self).__init__()
        self.in_channel = in_channel
        self.out_dim = out_dim
        self.audio_conv = nn.Sequential(
            SameBlock1d(in_channel, 128, kernel_size=7, padding=3),
            ResBlock1d(128, 128, 3, 1),
            # 9-5
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            # 5 -3
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            # 3-2
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, out_dim, kernel_size=3, padding=1),
        )

        self.global_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        ## b x c x t
        out = self.audio_conv(x)
        return self.global_avg(out).squeeze(2)


class SyncNet(nn.Module):
    """
    syncnet
    """

    # 15 29 128
    def __init__(self, in_channel_image, in_channel_audio, out_dim):
        super(SyncNet, self).__init__()
        self.in_channel_image = in_channel_image  # 15
        self.in_channel_audio = in_channel_audio  # 29 
        self.out_dim = out_dim
        self.face_encoder = FaceEncoder(in_channel_image, out_dim)
        self.audio_encoder = AudioEncoder(in_channel_audio, out_dim)
        self.merge_encoder = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim, 1, kernel_size=3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, image, audio):
        image_embedding = self.face_encoder(image)
        audio_embedding = (
            self.audio_encoder(audio)
            .unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, 1, image_embedding.size(2), image_embedding.size(3))
        )
        concat_embedding = torch.cat([image_embedding, audio_embedding], 1)
        out_score = self.merge_encoder(concat_embedding)
        out_score_normal = self.sigmoid(out_score)
        return out_score_normal


class SyncNetPerception(nn.Module):
    """
    use syncnet to compute perception loss
    """

    def __init__(self):
        super(SyncNetPerception, self).__init__()
        self.model = SyncNet(15, 29, 128)
        # print("load lip sync model : {}".format(pretrain_path))
        # self.model.load_state_dict(torch.load(pretrain_path)["state_dict"]["net"])
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # self.model.eval()

    def forward(self, image, audio):
        score = self.model(image, audio)
        return score
