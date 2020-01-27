import argparse
import os

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from data import DatasetFromFolder, SQLDataset, InfiniteSampler
from model import NMDiscriminator
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./root')
parser.add_argument('--val', type=str, default='./val')
parser.add_argument('--save_dir', type=str, default='./snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help="adam: learning rate")
parser.add_argument('--scale', type=int, default=4, help="scale")
parser.add_argument('--max_iter', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--resume', type=int)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cudnn.benchmark = True

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

writer = SummaryWriter()

train_set = SQLDataset(args.root)
iterator_train = iter(data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    sampler=InfiniteSampler(len(train_set)),
    num_workers=args.n_threads
))
print(len(train_set))

val_set = SQLDataset(args.val)
iterator_val = iter(data.DataLoader(
    val_set,
    batch_size=200,
    sampler=InfiniteSampler(len(val_set)),
    num_workers=args.n_threads
))
print(len(val_set))

nmd_model = NMDiscriminator().to(device)
bce = nn.BCELoss().to(device)

start_iter = 0
nmd_optimizer = torch.optim.Adam(
    nmd_model.parameters(),
    args.lr
)

if args.resume:
    nmd_checkpoint = torch.load(f'{args.save_dir}/ckpt/NMD_{args.resume}.pth', map_location=device)
    nmd_model.load_state_dict(nmd_checkpoint['model_state_dict'])
    print('Model restored!')
    start_iter = args.resume

    sigma = nmd_checkpoint['sigma']
    alpha = nmd_checkpoint['alpha']
else:
    sigma = 0.1
    alpha = 0.5

val_sigma_stack = [0.]*10
val_alpha_stack = [0.]*10
for i in tqdm(range(start_iter, args.max_iter)):
    _, target = [x.to(device) for x in next(iterator_train)]

    b = args.batch_size # fetch from target?
    noisy = get_noisy(target[:b//4], sigma)
    blurry = get_blurry(target[b//4:b//2], args.scale, alpha)
    clean = target[b//2:]
    input_train = torch.cat([noisy, blurry, clean])
    result = nmd_model(input_train)

    labels = torch.cat([
        torch.zeros((b//2, 1, 1, 1)),
        torch.ones((b//2, 1, 1, 1))
    ]).to(device)
    nmd_loss = torch.mean(bce(result, labels))

    nmd_optimizer.zero_grad()
    nmd_loss.backward()
    nmd_optimizer.step()

    if (i+1) % 100 == 0:
        _, val_target = [x.to(device) for x in next(iterator_val)]

        val_noisy = get_noisy(val_target[:100], sigma)    
        val_blurry = get_blurry(val_target[:100], args.scale, alpha)
        val_clean = val_target[100:]
        input_val_sigma = torch.cat([val_noisy, val_clean])
        input_val_alpha = torch.cat([val_blurry, val_clean])

        val_labels = torch.cat([
            torch.zeros((100, 1, 1, 1)),
            torch.ones((100, 1, 1, 1))
        ]).to(device)
        with torch.no_grad():
            sigma_acc = calc_acc(nmd_model(input_val_sigma), val_labels)
            alpha_acc = calc_acc(nmd_model(input_val_alpha), val_labels)
            train_acc = calc_acc(result, labels)

        val_sigma_stack.append(sigma_acc.item())
        val_alpha_stack.append(alpha_acc.item())
        val_sigma_stack = val_sigma_stack[1:]
        val_alpha_stack = val_alpha_stack[1:]
        sigma_avg = np.mean(val_sigma_stack)
        alpha_avg = np.mean(val_alpha_stack)

        if sigma_avg >= 95.:
            sigma = np.clip(sigma*.8, .0044, .1)
        if alpha_avg >= 95.:
            alpha = np.clip(alpha+.1, .5, .9)

    if (i+1) % args.save_model_interval == 0 or (i+1) == args.max_iter:
        torch.save({
            'model_state_dict': nmd_model.state_dict(),
            'sigma': sigma,
            'alpha': alpha,
            }, f'{args.save_dir}/ckpt/NMD_{i+1}.pth')

    if (i+1) % args.log_interval == 0:
        writer.add_scalar('nmd_loss', nmd_loss.item(), i+1)
        writer.add_scalar('train_acc', train_acc.item(), i+1)
        writer.add_scalar('sigma', sigma, i+1)
        writer.add_scalar('alpha', alpha, i+1)

        writer.add_scalar('sigma_avg', sigma_avg, i+1)
        writer.add_scalar('alpha_avg', alpha_avg, i+1)

        writer.add_scalar('val_sigma_acc', sigma_acc, i+1)
        writer.add_scalar('val_alpha_acc', alpha_acc, i+1)

writer.close()
