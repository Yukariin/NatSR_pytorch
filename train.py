import argparse
import os

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from data import SQLDataset, InfiniteSampler
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./root')
parser.add_argument('--save_dir', type=str, default='./snapshots')
parser.add_argument('--lr', type=float, default=2e-4, help="adam: learning rate")
parser.add_argument('--l1', type=float, default=1., help="lambda1")
parser.add_argument('--l2', type=float, default=1e-3, help="lambda2")
parser.add_argument('--l3', type=float, default=1e-3, help="lambda3")
parser.add_argument('--scale', type=int, default=4, help="scale")
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--resume', type=int)
parser.add_argument('--nmd', type=str)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cudnn.benchmark = True

if not os.path.exists(args.save_dir):
    os.makedirs(f'{args.save_dir}/ckpt')

writer = SummaryWriter()

train_set = SQLDataset(args.root)
iterator_train = iter(data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    sampler=InfiniteSampler(len(train_set)),
    num_workers=args.n_threads
))
print(len(train_set))

g_model = NSRNet(scale=args.scale).to(device)
d_model = Discriminator().to(device)
nmd_model = NMDiscriminator().to(device)
bce_s = nn.BCEWithLogitsLoss().to(device)
l1 = nn.L1Loss().to(device)

start_iter = 0
g_optimizer = torch.optim.Adam(
    g_model.parameters(),
    args.lr)
d_optimizer = torch.optim.Adam(
    d_model.parameters(),
    args.lr)

if args.resume:
    g_checkpoint = torch.load(f'{args.save_dir}/ckpt/G_{args.resume}.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)
    d_checkpoint = torch.load(f'{args.save_dir}/ckpt/D_{args.resume}.pth', map_location=device)
    d_model.load_state_dict(d_checkpoint)

    print('Model loaded!')
    start_iter = args.resume

nmd_checkpoint = torch.load(args.nmd, map_location=device)
nmd_model.load_state_dict(nmd_checkpoint['model_state_dict'])
nmd_model.eval()

for i in tqdm(range(start_iter, args.max_iter)):
    input, target = [x.to(device) for x in next(iterator_train)]

    result = g_model(input)


    #  Train D
    y_pred_fake = d_model(result.detach())
    y_pred = d_model(target)
    y = torch.ones_like(y_pred)
    y2 = torch.zeros_like(y_pred)
    d_loss = (bce_s(y_pred - torch.mean(y_pred_fake), y) + bce_s(y_pred_fake - torch.mean(y_pred), y2))/2

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()


    #  Train G
    recon_loss = l1(result, target)

    n_pred = nmd_model(result)
    nat_loss = torch.mean(-torch.log(n_pred+1e-10))
    nat_score = torch.mean(n_pred)

    y_pred_fake = d_model(result)
    y_pred = d_model(target)
    y = torch.ones_like(y_pred)
    y2 = torch.zeros_like(y_pred)
    g_loss = (bce_s(y_pred - torch.mean(y_pred_fake), y2) + bce_s(y_pred_fake - torch.mean(y_pred), y))/2

    total_loss = args.l1*recon_loss + args.l2*nat_loss + args.l3*g_loss

    g_optimizer.zero_grad()
    total_loss.backward()
    g_optimizer.step()


    if (i+1) % args.save_model_interval == 0 or (i+1) == args.max_iter:
        torch.save(g_model.state_dict(), f'{args.save_dir}/ckpt/G_{i+1}.pth')
        torch.save(d_model.state_dict(), f'{args.save_dir}/ckpt/D_{i+1}.pth')

    if (i+1) % args.log_interval == 0:
        writer.add_scalar('recon_loss', recon_loss.item(), i+1)
        writer.add_scalar('total_loss', total_loss.item(), i+1)
        writer.add_scalar('nat_loss', nat_loss.item(), i+1)
        writer.add_scalar('nat_score', nat_score.item(), i+1)
        writer.add_scalar('g_loss', g_loss.item(), i+1)
        writer.add_scalar('d_loss', d_loss.item(), i+1)

    if (i+1) % args.vis_interval == 0:
        ims = torch.cat([target, result], dim=3)
        grid = make_grid(ims[:4], 2).clamp_(0,1)
        writer.add_image('target_result', grid, i+1)
        writer.flush()

writer.close()
