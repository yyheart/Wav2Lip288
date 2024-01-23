from os.path import dirname, join, basename, isfile
from tqdm import tqdm
import torch.autograd as autograd

from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual
import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob
import torch.multiprocessing as mp
import torch.distributed as dist
import os, random, cv2, argparse
from hparams import hparams, get_image_list_new


parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed  dataset", default="/mnt/sdb/liwen/data/lrs2_preprocessed",required=False, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default="/mnt/sdb/liwen/wav2lip_288x288/checkpoints/wloss",required=False, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', default="/mnt/sdb/liwen/wav2lip_288x288/checkpoints/sync_original/lipsync_expert.pth",required=False, type=str)
parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

LAMBDA = 10  # gradient penalty


def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss = 0., 0.
        running_gp = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            disc.train()
            model.train()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            ### Train generator now. Remove ALL grads. 
            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            g = model(indiv_mels, x)

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            if hparams.disc_wt > 0.:
                pred = disc(g)
                perceptual_loss = -pred.mean()
            else:
                perceptual_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                    (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

            loss.backward()
            optimizer.step()

            ### Remove all gradients before Training disc
            disc_optimizer.zero_grad()

            pred = disc(gt)
            disc_real_loss = -pred.mean()
            
            fake_img = g.detach()
            pred = disc(fake_img)
            disc_fake_loss = pred.mean()

            disc_fake_loss.backward()

            # gradient penalty
            alpha = torch.rand(1)*torch.ones(gt.size(0), 1)
            alpha = alpha.expand(gt.size(0), int(gt.nelement()/gt.size(0))).contiguous().view(gt.size(0), 3, 5, 288, 288).to(device)
            interpolates = alpha * gt + ((1 - alpha) * fake_img)
            interpolates = interpolates.to(device)
            interpolates = autograd.Variable(interpolates, requires_grad=True)

            disc_interpolates = disc(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)

            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
            gradient_penalty.backward()

            disc_optimizer.step()

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()
            running_gp += gradient_penalty.item()
            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            # Logs
            global_step += 1
            cur_session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if hparams.disc_wt > 0.:
                running_perceptual_loss += perceptual_loss.item()
            else:
                running_perceptual_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
                save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, global_epoch, prefix='disc_')


            if global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, disc)

                    if average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.03)

            prog_bar.set_description('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(running_l1_loss / (step + 1),
                                                                                        running_sync_loss / (step + 1),
                                                                                        running_perceptual_loss / (step + 1),
                                                                                        running_disc_fake_loss / (step + 1),
                                                                                        running_disc_real_loss / (step + 1)))

        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, disc):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
    while 1:
        for step, (x, indiv_mels, mel, gt) in enumerate((test_data_loader)):
            model.eval()
            disc.eval()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

            g = model(indiv_mels, x)
            pred = disc(g)
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            sync_loss = get_sync_loss(mel, g)
            
            if hparams.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                    (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())
            
            if hparams.disc_wt > 0.:
                running_perceptual_loss.append(perceptual_loss.item())
            else:
                running_perceptual_loss.append(0.)

            if step > eval_steps: break

        print('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(sum(running_l1_loss) / len(running_l1_loss),
                                                            sum(running_sync_loss) / len(running_sync_loss),
                                                            sum(running_perceptual_loss) / len(running_perceptual_loss),
                                                            sum(running_disc_fake_loss) / len(running_disc_fake_loss),
                                                             sum(running_disc_real_loss) / len(running_disc_real_loss)))
        return sum(running_sync_loss) / len(running_sync_loss)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train_lrs2_remove')
    test_dataset = Dataset('val_lrs2_remove')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

     # Model
    model = Wav2Lip().to(device)
    disc = Wav2Lip_disc_qual().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer, 
                                reset_optimizer=False, overwrite_global_states=False)
        
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, 
                                overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
