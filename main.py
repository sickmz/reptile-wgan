import os
from args import get_args
import numpy as np
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, ConcatDataset
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchvision import datasets, transforms
from models import Generator, Discriminator
from metatrain import initialize_meta_optimizer, train_meta_learner, test_meta_learner, gradient_penalty
from collections import defaultdict

SAMPLE_SIZE = 80
NUM_DATASET = 2
NUM_FULL_LOADER_DATASET = 1

finetuning_model = False
loaded_model = False

# Parsing params
args = get_args()

# Convert the range from [-1, 1] to [0, 1]
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

# Single loader from each dataset
def multiLoaders(args, data_list):
    return DataLoader(data_list, sampler = RandomSampler(data_list, replacement=True), batch_size=args.batch_size, num_workers=64)

# Runs well in single thread, but may be crashed in multiple onesdef makeLoader(args, data_list):
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(args.samples_dir):
    os.mkdir(args.samples_dir)   
if not os.path.exists(args.output):
    os.mkdir(args.output)

transform = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Prepare data
trainset0 = datasets.CIFAR10(root='./data/', train=True, download=False, transform=transform)
trainloader0 = torch.utils.data.DataLoader(trainset0, batch_size=args.batch_size, shuffle=True, num_workers=2)
trainset1 = datasets.CelebA('./data/celeba', transform=transform)
trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=args.batch_size, shuffle=True, num_workers=2)

current_loader = None
fullLoader = None 

# Create N single dataset loaders
loaders = [multiLoaders(args, x) for x in (trainset0, trainset1)]

model_d = Discriminator()
model_g = Generator()

# Generate fixed noise 
fixed_noise = torch.FloatTensor(SAMPLE_SIZE, args.nz, 1, 1).normal_(0, 1)

# Send model to the current device
model_d.cuda()
model_g.cuda()
fixed_noise = Variable(fixed_noise).cuda()

meta_iteration = 0

# Generate random sample of singles loader from each dataset
for n in range(NUM_DATASET - NUM_FULL_LOADER_DATASET):
    print("sample: generated from dataset: {}".format(eval("trainset"+str(n)).__class__.__name__))
    imgs,_ = next(iter(loaders[n]))
    
    filename = '{}/{}_{}.png'.format(args.samples_dir, n, eval("trainset"+str(n)).__class__.__name__)
    save_image(denorm(imgs), filename)

# Start meta-training
if not finetuning_model:

    # Training
    for epoch_idx in range(args.epochs):
        
        model_d.train()
        model_g.train()

        # Dataset choice for training
        rdataset, floader = random.sample([0, 1], NUM_DATASET)

        current_loader = loaders[rdataset]
        fullLoader = loaders[floader]

        meta_optimizer_d = initialize_meta_optimizer(model_d)
        meta_optimizer_g = initialize_meta_optimizer(model_g)

        cloned_d = model_d.clone()
        cloned_g = model_g.clone()

        (errD_real, errD_fake, fakeD_mean, realD_mean), (errG) = \
            train_meta_learner(
                model_d, model_g, cloned_d, cloned_g, meta_optimizer_d, meta_optimizer_g, fullLoader,
                current_loader,
                meta_iteration, args.epochs
        )

        meta_iteration += 1

        errD = errD_real + errD_fake
        d_loss = errD.item()
        g_loss = errG.item()

        # Test model
        if  (epoch_idx + 1) % args.save_every == 0:
            print(
                "\t({} / {}) mean D(fake) = {:.4f}/{:.4f}, mean D(real) = {:.4f}".format(
                    epoch_idx, args.epochs,
                    fakeD_mean, errG, realD_mean
                )
            )

            for num in range(NUM_DATASET - NUM_FULL_LOADER_DATASET): 

                test_current_loader = loaders[num]

                # Update the model on test label
                g_out = test_meta_learner(
                    model_g, model_d, fullLoader, test_current_loader, fixed_noise)

                filename = '{}/{}_{}.png'.format(args.output, epoch_idx, num)
                save_image(denorm(g_out.detach()), filename)

            # Save all model
            torch.save(model_g, args.save_dir + "/G_" + str(epoch_idx) + ".pth")
            torch.save(model_d, args.save_dir + "/D_" + str(epoch_idx) +".pth")

        print('Epoch {}     D loss = {:.4f}     G loss = {:.4f}'.format(
            epoch_idx, d_loss, g_loss
        ))

# WIP
if finetuning_model:

    # Load model to be finetuned
    if(loaded_model == True):
        print("Meta Model")
        model_g = torch.load(args.trans_save_dir + "/G.pth").cuda()
        model_d = torch.load(args.trans_save_dir + "/D.pth").cuda()
    else:
        print("Model from scratch")
        model_d = Discriminator().cuda()
        model_g = Generator().cuda()

    optimizerD = torch.optim.Adam(model_d.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = torch.optim.Adam(model_g.parameters(), lr=1e-4, betas=(0.5, 0.9))

    noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1).cuda()
    one_hot_labels = torch.FloatTensor(args.batch_size, 10).cuda()

    for epoch_idx in range(args.ft_epochs):
        
        for i in range(5):
            x,_ = next(iter(trainloader))

            # Train Discriminator
            x = x.cuda()
            inputv = Variable(x)

        # Discriminator training

            # Real examples
            output,_ = model_d(inputv)
            errD_real = -output.mean()
            realD_mean = output.data.cpu().mean()

            noise.resize_(args.batch_size, args.nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)

            # Fake examples
            g_out = model_g(noisev)
            output,_ = model_d(g_out.detach())
            errD_fake =  output.mean() 
            fakeD_mean = output.data.cpu().mean()

            # Compute loss for gradient penalty.
            alpha = torch.rand(x.size(0), 1, 1, 1).cuda()
            x_hat = (alpha * x.data + (1 - alpha) * g_out.data).requires_grad_(True)
            out_src,_ = model_d(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat)

            loss =  errD_real + errD_fake + 10*d_loss_gp
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            loss.backward()

            # Next
            optimizerD.step()

        one_hot_labels.zero_()      
        noise.resize_(args.batch_size, args.nz, 1, 1).normal_(0, 1)

        noisev = Variable(noise)

        # Generator training
        g_out = model_g(noisev)
        output, _ = model_d(g_out)
        err = - output.mean()
        optimizerG.zero_grad()
        optimizerD.zero_grad()
        err.backward()
        errG = err.data.cpu()

        # Next
        optimizerG.step()

        print('Epoch {} - D real loss = {:.4f}, D fake loss = {:.4f} G loss = {:.4f}'.format(
            epoch_idx, realD_mean, fakeD_mean, errG
        ))

        if epoch_idx % 1000 == 0:
            print("Save images...")
            g_out = model_g(fixed_noise)

            filename = '{}/{}.png'.format(args.trans_samples_dir, epoch_idx)
            save_image(denorm(g_out.detach()), filename)

            torch.save(model_g, args.trans_save_dir + "/no_meta_G_finetuned_no_meta_only_D_100.pth")
            torch.save(model_d, args.trans_save_dir + "/no_meta_D_finetuned_no_meta_only_D_100.pth")
