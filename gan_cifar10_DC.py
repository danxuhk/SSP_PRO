import os, sys
sys.path.append(os.getcwd())

import time
import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10_ori_all
import tflib.plot
import tflib.inception_score

import numpy as np


import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = 'cifar-10-batches-py/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)
inception_score_all = [];
results_save = './results_new/dc_1'
if not os.path.isdir(results_save):
        os.makedirs(results_save);

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm2d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

netG = Generator()
netD1 = Discriminator()
netD2 = Discriminator()
print netG
print netD1

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 9
if use_cuda:
    netD1 = netD1.cuda(gpu)
    netD2 = netD2.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD1 = optim.Adam(netD1.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD2 = optim.Adam(netD2.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# For generating samples
def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, (results_save + '/samples_{}.jpg').format(frame))

# For calculating inception score
def get_inception_score(G, ):
    all_samples = []
    for i in xrange(10):
        samples_100 = torch.randn(100, 128)
        if use_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterator
train_gen, dev_gen = lib.cifar10_ori_all.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images, target in train_gen():
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images
gen = inf_train_gen()
preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

for iteration in xrange(ITERS):
    start_time = time.time()
	
    ############################
    # (1) Update D networks
    ###########################
    for p in netD1.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for p in netD2.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
		
	
    for i in xrange(CRITIC_ITERS):

		#netD1.zero_grad() # this should be here or later?
		#netD2.zero_grad()
		
		data1 = []
		data2 = []

		# split data according to dicriminative clustering ----------------------
		for v in xrange(2):
			_data = gen.next()
			_data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
			real_data = torch.stack([preprocess(item) for item in _data])

			if use_cuda:
				real_data = real_data.cuda(gpu)
			r_data_v = autograd.Variable(real_data)

			D1 = netD1(r_data_v)
			scores1 = D1.cpu().data.numpy().flatten()		
			
			D2 = netD2(r_data_v)
			scores2 = D2.cpu().data.numpy().flatten()		
					
			for d in xrange(BATCH_SIZE):
				if (scores1[d] > scores2[d] and data1.len() < BATCH_SIZE) or data2.len() >= BATCH_SIZE:
                                    data1.append(_data[d, :, :, :])
                                else:
                                    data2.append(_data[d, :, :, :])
								
		# ------------------------------------------------------
			
		
		# update D1 with real data -----------------------------
		
		netD1.zero_grad()
		
		data1 = data1.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
		real_data1 = torch.stack([preprocess(item) for item in data1])

		if use_cuda:
			real_data1 = real_data1.cuda(gpu)
		real_data_v1 = autograd.Variable(real_data1)

		D_real1 = netD1(real_data_v1)
		D_real1 = D_real1.mean()
		D_real1.backward(mone)		
			
		# update D2 with real data -----------------------------
		
		netD2.zero_grad()
		
		data2 = data2.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
		real_data2 = torch.stack([preprocess(item) for item in data2])

		if use_cuda:
			real_data2 = real_data2.cuda(gpu)
		real_data_v2 = autograd.Variable(real_data2)

		D_real2 = netD2(real_data_v2)
		D_real2 = D_real2.mean()
		D_real2.backward(mone)		


        # train with fake -----------------------------
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv1 = fake
        inputv2 = fake
		
        D_fake1 = netD1(inputv1)
        D_fake1 = D_fake1.mean()
        D_fake1.backward(one)

        D_fake2 = netD2(inputv2)
        D_fake2 = D_fake2.mean()
        D_fake2.backward(one)
		
        # train with gradient penalty -----------------------------
		
        gradient_penalty1 = calc_gradient_penalty(netD1, real_data_v1.data, fake.data) # should I use fake1 and fake2 ???
        gradient_penalty1.backward()

        gradient_penalty2 = calc_gradient_penalty(netD2, real_data_v2.data, fake.data)
        gradient_penalty2.backward()
		
		
        # print "gradien_penalty: ", gradient_penalty

		D_cost = 0.5 * (D_fake1 - D_real1 + gradient_penalty1) + 0.5 * (D_fake2 - D_real2 + gradient_penalty2)
		
		Wasserstein_D = 0.5 * (D_real1 - D_fake1) + 0.5 * (D_real2 - D_fake2)

        optimizerD1.step()
        optimizerD2.step()
		
		
		
    ############################
    # (2) Update G network
    ###########################
    for p in netD1.parameters():
        p.requires_grad = False  # to avoid computation
    for p in netD2.parameters():
        p.requires_grad = False  # to avoid computation

	netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev) # should i use fake1 and fake2 ???
	
    G1 = netD1(fake)
    G1 = G1.mean()
    G1.backward(mone)
    G_cost1 = -G1
    optimizerG.step()

    G2 = netD2(fake)
    G2 = G2.mean()
    G2.backward(mone)
    G_cost2 = -G2
    optimizerG.step()


    # Write logs and save samples
    lib.plot.plot(results_save + '/train disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot(results_save + '/time', time.time() - start_time)
    lib.plot.plot(results_save + '/train gen cost', 0.5 * G_cost1.cpu().data.numpy() + 0.5 * G_cost2.cpu().data.numpy())
    lib.plot.plot(results_save + '/wasserstein distance', Wasserstein_D.cpu().data.numpy())

    # Calculate inception score every 1K iters
    if False and iteration % 25000 == 24999:
        inception_score = get_inception_score(netG)
        inception_score_all = np.append(inception_score_all, inception_score[0]);
        lib.plot.plot(results_save + '/inception score', inception_score[0])

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 100 == 99:
        dev_disc_costs = []
        for images, _ in dev_gen():
            images = images.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            imgs = torch.stack([preprocess(item) for item in images])

            # imgs = preprocess(images)
            if use_cuda:
                imgs = imgs.cuda(gpu)
            imgs_v = autograd.Variable(imgs, volatile=True)

            D1 = netD1(imgs_v) # this can also be avoided if it is time consuming...
            D2 = netD2(imgs_v)
			
            _dev_disc_cost = 0.5 * (-D1.mean().cpu().data.numpy()) + 0.5 * (-D2.mean().cpu().data.numpy())
            dev_disc_costs.append(_dev_disc_cost)
        lib.plot.plot(results_save + '/dev disc cost', np.mean(dev_disc_costs))

        generate_image(iteration, netG)

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()
    lib.plot.tick()
print inception_score_all
np.savetxt(results_save + '/inception_score.txt', inception_score_all, delimiter=',');
