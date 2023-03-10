__author__ = "Keenan Manpearl"
__date__ = "2023/03/01"

"""
Training a td-vae model on mitocheck movies
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pathlib
import pandas as pd
import sys
import os
from torch.utils.data import DataLoader, Dataset
from model import TD_VAE, DBlock, PreProcess, Decoder

sys.path.append(os.path.abspath("./"))
from prep_data import Mitocheck_Dataset

log_file = pathlib.Path("loginfo_compression10.txt")
data_file = pathlib.Path("mitocheck_compression10_2movies.pkl")

with open(data_file, "rb") as file_handle:
    mitocheck = pickle.load(file_handle)
data = Mitocheck_Dataset(mitocheck)


# Set hyperparameters
# all hyperparameters taken from original paper
batch_size = 2
num_epoch = 1
learning_rate = 0.00005
input_size = 102 * 134
processed_x_size = 102 * 134
belief_state_size = 50
state_size = 8
d_block_hidden_size = 50
decoder_hidden_size = 200

data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Build a TD-VAE model
print("making td-vae")
tdvae = TD_VAE(
    x_size=input_size,
    processed_x_size=processed_x_size,
    b_size=belief_state_size,
    z_size=state_size,
    d_block_hidden_size=d_block_hidden_size,
    decoder_hidden_size=decoder_hidden_size,
)


tdvae = tdvae.cuda()

print("making optimizer")
optimizer = optim.Adam(tdvae.parameters(), lr=learning_rate)
z_values = []

print("training model")
with open(log_file, "w") as log_file_handle:
    for epoch in range(num_epoch):
        for idx, images in enumerate(data_loader):
            images = images.cuda()
            # Make a forward step of preprocessing and LSTM
            for t in range(9):
                z = tdvae.forward(images, t)
                z_values.append(z.cpu().detach().numpy())
# print(np.array(z_values).shape)


z_df = pd.DataFrame(columns=["x", "y", "z", "vid"])
for i in range(9):
    for j in range(2):
        for k in range(16):
            # frame1 = z_values[i][0]
            x = i
            z = z_values[i][j][k]
            y = k
            vid = j
            row = pd.DataFrame({"x": [x], "y": [y], "z": [z], "vid": [vid]})
            z_df = pd.concat([z_df, row])
print(z_df)
# vid1.append(frame1)
# frame2 = z_values[i][1]
# vid2.append(frame2)


fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(projection="3d")

colors = {0: "orange", 1: "g"}
for i in range(2):
    ax.scatter(
        xs=z_df.loc[z_df.vid == i, "x"],
        ys=z_df.loc[z_df.vid == i, "y"],
        zs=z_df.loc[z_df.vid == i, "z"],
        color=colors[i],
    )

plt.savefig("latent_space.png")
plt.show()


# z = [vid1, vid2]
# output = open("z_values.pkl", "wb")
# pickle.dump(z, output)
# output.close()
