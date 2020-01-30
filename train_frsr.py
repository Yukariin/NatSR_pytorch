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
parser.add_argument('--scale', type=int, default=4, help="scale")
parser.add_argument('--max_iter', type=int, default=120000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--resume', type=int)
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
l1 = nn.L1Loss().to(device)

start_iter = 0
g_optimizer = torch.optim.Adam(
    g_model.parameters(),
    args.lr)

if args.resume:
    g_checkpoint = torch.load(f'{args.save_dir}/ckpt/G_{args.resume}.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)

    print('Model loaded!')
    start_iter = args.resume

for i in tqdm(range(start_iter, args.max_iter)):
    input, target = [x.to(device) for x in next(iterator_train)]

    result = g_model(input)
    
    recon_loss = l1(result, target)
    
    g_optimizer.zero_grad()
    recon_loss.backward()
    g_optimizer.step()

    if (i+1) % args.save_model_interval == 0 or (i+1) == args.max_iter:
        torch.save(g_model.state_dict(), f'{args.save_dir}/ckpt/G_{i+1}.pth')

    if (i+1) % args.log_interval == 0:
        writer.add_scalar('recon_loss', recon_loss.item(), i+1)

    if (i+1) % args.vis_interval == 0:
        ims = torch.cat([target, result], dim=3)
        grid = make_grid(ims[:4], 2).clamp_(0,1)
        writer.add_image('target_result', grid, i+1)
        writer.flush()

writer.close()
