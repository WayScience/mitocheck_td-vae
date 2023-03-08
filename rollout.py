__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>
After training the model, we can try to use the model to do jumpy predictions.
"""

import cv2
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from prep_data import Mitocheck_Dataset
from model import TD_VAE, DBlock, PreProcess, Decoder


#### load trained model
checkpoint = torch.load("output/compression10_epoch_49.pt")
input_size = 102*134
processed_x_size = 102*134
belief_state_size = 50
state_size = 8
d_block_hidden_size = 50
decoder_hidden_size = 200

tdvae = TD_VAE(
    x_size=input_size,
    processed_x_size=processed_x_size,
    b_size=belief_state_size,
    z_size=state_size,
    d_block_hidden_size=d_block_hidden_size,
    decoder_hidden_size=decoder_hidden_size,
)

optimizer = optim.Adam(tdvae.parameters(), lr=0.00005)

tdvae.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

data_file = "mitocheck_compression10_2movies.pkl"
with open(data_file, "rb") as file_handle:
    mitocheck = pickle.load(file_handle)

tdvae.eval()
tdvae = tdvae.cuda()

data = Mitocheck_Dataset(mitocheck)
batch_size = 2
data_loader = DataLoader(data, batch_size = batch_size, shuffle=True)
idx, images = next(enumerate(data_loader))

images = images.cuda()

## calculate belief
tdvae.forward(images)

## jumpy rollout
t1, t2 = 1, 3
rollout_images = tdvae.rollout(images, t1, t2)

#### plot results
fig = plt.figure(0, figsize=(12, 4))

fig.clf()
gs = gridspec.GridSpec(batch_size, t2 + 2)
gs.update(wspace=0.05, hspace=0.05)
for i in range(batch_size):
    for j in range(t1):
        axes = plt.subplot(gs[i, j])
        axes.imshow(1 - images.cpu().data.numpy()[i, j].reshape(102, 134), cmap="binary")
        axes.axis("off")

    for j in range(t1, t2 + 1):
        axes = plt.subplot(gs[i, j + 1])
        axes.imshow(
            1 - rollout_images.cpu().data.numpy()[i, j - t1].reshape(102, 134),
            cmap="binary",
        )
        axes.axis("off")

fig.savefig("./output/rollout_compression10_49.eps")
plt.show()
sys.exit()
