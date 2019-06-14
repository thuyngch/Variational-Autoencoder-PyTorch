#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from shutil import rmtree

import torch
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


#------------------------------------------------------------------------------
#  Arguments
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs', type=int, default=200,
					help='number of epochs of training')

parser.add_argument('--batch_size', type=int, default=128,
					help='size of the batches')

parser.add_argument('--lr', type=float, default=5e-4,
					help='adam: learning rate')

parser.add_argument('--n_cpus', type=int, default=8,
					help='number of cpu threads to use during batch generation')

parser.add_argument('--log_dir', type=str, default="./logging",
					help='use cuda to train model')

args = parser.parse_args()


#------------------------------------------------------------------------------
#  Setup
#------------------------------------------------------------------------------
# Initialize VAE
model = VAEGT(in_dims=784, hid_dims=100, num_classes=10)
model.cuda()

# Configure data loader
data_dir = "/home/cybercore/thuync/datasets/"
os.makedirs(data_dir, exist_ok=True)
dataset = datasets.MNIST(data_dir, train=True, download=True,
	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
]))
dataloader = torch.utils.data.DataLoader(
	dataset, batch_size=args.batch_size,
	num_workers=args.n_cpus, shuffle=True, pin_memory=True,
)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# TensorboardX
if os.path.exists(args.log_dir):
	rmtree(args.log_dir)
os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.log_dir)

# ImproveChecker
improvechecker = ImproveChecker(mode='min')


#------------------------------------------------------------------------------
#  Training
#------------------------------------------------------------------------------
model.train()
for epoch in range(1, args.n_epochs+1):
	for i, (imgs, gts) in enumerate(dataloader):
		# Prepare input
		inputs = imgs.view(imgs.shape[0], -1).cuda()
        gts = gts.view(-1, 1).cuda()

		# Train
		optimizer.zero_grad()
		outputs, mu, logvar = model(inputs, gts)
		loss = loss_fn(outputs, inputs, mu, logvar)
		loss.backward()
		optimizer.step()

	# Logging
	outputs = outputs.view(-1, 1, 28, 28)
	grid = make_grid(outputs.data[:25], nrow=5, normalize=True)
	writer.add_image('output', grid, epoch)
	writer.add_scalar("loss", loss.item(), epoch)
	print("[EPOCH %.3d] Loss: %.6f" % (epoch, loss.item()))

	# ImproveChecker
	if improvechecker.check(loss.item()):
		checkpoint = dict(
			epoch=epoch,
			loss=loss.item(),
			state_dict=model.state_dict(),
			optimizer=optimizer.state_dict(),
		)
		save_file = os.path.join(LOG_DIR, "best_checkpoint.pth")
		torch.save(checkpoint, save_file)
		print("Best checkpoint is saved at %s" % (save_file))
