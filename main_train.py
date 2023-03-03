__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 16:45:38"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pathlib
from torch.utils.data import DataLoader, Dataset
from prep_data import Mitocheck_Dataset
from model import TD_VAE, DBlock, PreProcess, Decoder
import matplotlib.pyplot as plt


log_file = pathlib.Path("loginfo.txt")
data_file = pathlib.Path("mitocheck_4_plates.pkl")

with open(data_file, "rb") as file_handle:
    mitocheck = pickle.load(file_handle)
data = Mitocheck_Dataset(mitocheck)

# Set constants
time_constant_max = 6  # There are 9 frames total
time_jump_options = [1, 2, 3]  # Jump up to 3 frames away

# Set hyperparameters
batch_size = 20
num_epoch = 1000
learning_rate = 0.0005

data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Build a TD-VAE model
input_size = 3300
processed_x_size = 3300
belief_state_size = 50
state_size = 8 
d_block_hidden_size = 50
decoder_hidden_size = 200

tdvae = TD_VAE(
    x_size=input_size,
    processed_x_size=processed_x_size,
    b_size=belief_state_size,
    z_size=state_size,
    d_block_hidden_size = d_block_hidden_size,
    decoder_hidden_size = decoder_hidden_size
)


tdvae = tdvae.cuda()

# Train model
optimizer = optim.Adam(tdvae.parameters(), lr=learning_rate)

with open(log_file, "w") as log_file_handle:
    for epoch in range(num_epoch):
        for idx, images in enumerate(data_loader):
            images = images.cuda()

            # Make a forward step of preprocessing and LSTM
            tdvae.forward(images)

            # Randomly sample a time step and jumpy step
            t_1 = np.random.choice(time_constant_max)
            t_2 = t_1 + np.random.choice(time_jump_options)

            # Calculate loss function based on two time points
            loss = tdvae.calculate_loss(t_1, t_2)

            # must clear out stored gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(
                    epoch, idx, loss.item()
                ),
                file=log_file_handle,
                flush=True,
            )

            print(
                "epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(
                    epoch, idx, loss.item()
                )
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": tdvae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                f"output/model_epoch_{epoch}.pt",
            )