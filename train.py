import argparse
import os

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from data import DatasetFromFolder, InfiniteSampler
from model import NSRNet, Discriminator, NMDiscriminator
from utils import get_manifold


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./root')
parser.add_argument('--save_dir', type=str, default='./snapshots')
parser.add_argument('--lr', type=float, default=2e-4, help="adam: learning rate")
parser.add_argument('--l1', type=float, default=1., help="lambda1")
parser.add_argument('--l2', type=float, default=1e-3, help="lambda2")
parser.add_argument('--l3', type=float, default=1e-3, help="lambda3")
parser.add_argument('--scale', type=int, default=4, help="scale")
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--resume', type=int)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cudnn.benchmark = True

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

writer = SummaryWriter()

size = (args.image_size, args.image_size)
input_tf = transforms.Compose([
    transforms.CenterCrop(args.image_size),
    transforms.Resize(args.image_size // args.scale),
    transforms.ToTensor(),
])
target_tf = transforms.Compose([
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
])

train_set = DatasetFromFolder(args.root, input_tf, target_tf)
iterator_train = iter(data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    sampler=InfiniteSampler(len(train_set)),
    num_workers=args.n_threads
))
print(len(train_set))

g_model = NSRNet().to(device)
d_model = Discriminator().to(device)
nmd_model = NMDiscriminator().to(device)
bce = nn.BCELoss().to(device)
bce_s = nn.BCEWithLogitsLoss().to(device)
l1 = nn.L1Loss().to(device)

start_iter = 0
g_optimizer = torch.optim.Adam(
    g_model.parameters(),
    args.lr)
d_optimizer = torch.optim.Adam(
    d_model.parameters(),
    args.lr)
nmd_optimizer = torch.optim.Adam(
    nmd_model.parameters(),
    args.lr)

if args.resume:
    g_checkpoint = torch.load(f'{args.save_dir}/ckpt/G_{args.resume}.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)
    d_checkpoint = torch.load(f'{args.save_dir}/ckpt/D_{args.resume}.pth', map_location=device)
    d_model.load_state_dict(d_checkpoint)
    nmd_checkpoint = torch.load(f'{args.save_dir}/ckpt/NMD_{args.resume}.pth', map_location=device)
    nmd_model.load_state_dict(nmd_checkpoint)
    print('Model restored!')
    start_iter = args.resume

alpha = 0.5
sigma = 0.1
for i in tqdm(range(start_iter, args.max_iter)):
    input, target = [x.to(device) for x in next(iterator_train)]

    result = g_model(input)

    recon_loss = l1(result, target)

    a, b = get_manifold(target, args.scale, alpha, sigma)
    a_b = torch.cat([a, b], 0)
    un_man = nmd_model(a_b)
    n_man = nmd_model(target)
    n_pred = nmd_model(result)
    y = torch.ones_like(un_man)
    y2 = torch.zeros_like(n_man)
    nmd_loss = (bce(un_man, y) + bce(n_man, y2))/2
    nat_loss = torch.mean(-torch.log(n_pred))

    y_pred_fake = d_model(result)
    y_pred = d_model(target)
    y = torch.ones_like(y_pred)
    y2 = torch.zeros_like(y_pred)
    g_loss = (bce_s(y_pred - torch.mean(y_pred_fake), y2) + bce_s(y_pred_fake - torch.mean(y_pred), y))/2
    d_loss = (bce_s(y_pred - torch.mean(y_pred_fake), y) + bce_s(y_pred_fake - torch.mean(y_pred), y2))/2

    total_loss = args.l1*recon_loss + args.l2*nat_loss + args.l3*g_loss

    g_optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    g_optimizer.step()

    d_optimizer.zero_grad()
    d_loss.backward(retain_graph=True)
    d_optimizer.step()

    nmd_optimizer.zero_grad()
    nmd_loss.backward()
    nmd_optimizer.step()
    
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        torch.save(g_model.state_dict(), f'{args.save_dir}/ckpt/G_{i + 1}.pth')
        torch.save(d_model.state_dict(), f'{args.save_dir}/ckpt/D_{i + 1}.pth')
        torch.save(nmd_model.state_dict(), f'{args.save_dir}/ckpt/NMD_{i + 1}.pth')

    if (i + 1) % args.log_interval == 0:
        writer.add_scalar('g_loss/recon_loss', recon_loss.item(), i + 1)
        writer.add_scalar('g_loss/g_loss', g_loss.item(), i + 1)
        writer.add_scalar('g_loss/nat_loss', nat_loss.item(), i + 1)
        writer.add_scalar('g_loss/total_loss', total_loss.item(), i + 1)
        writer.add_scalar('d_loss/d_loss', d_loss.item(), i + 1)
        writer.add_scalar('d_loss/nmd_loss', nmd_loss.item(), i + 1)

    if (i + 1) % args.vis_interval == 0:
        ims = torch.cat([target, result], dim=3)
        writer.add_images('target_result', ims, i + 1)

writer.close()