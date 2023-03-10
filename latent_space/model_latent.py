__author__ = "Keenan Manpearl"
__date__ = "2023/01/24"

"""
Classes and functions needed to train model
"""

import numpy as np
import torch
import torch.nn as nn


class DBlock(nn.Module):
    """A basic building block for parameterizing a normal distribution.
    It corresponds to the D operation in the reference Appendix.

    Returns a mean and log sigma for any arbitrary context X
    [mu, log_sigma] = W3*tanh(W1*x + B1)*sigmoid(W2*x + B2) + B3
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Construct fully connected linear layers to map
        # X to a normal distribution
        self.weight1 = nn.Linear(input_size, hidden_size)
        self.weight2 = nn.Linear(input_size, hidden_size)
        self.mu = nn.Linear(hidden_size, output_size)
        self.logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.weight1(input))
        t = t * torch.sigmoid(self.weight2(input))
        mu = self.mu(t)
        logsigma = self.logsigma(t)

        return mu, logsigma


class PreProcess(nn.Module):
    """
    The pre-process layer for MNIST image
    """

    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t


class Decoder(nn.Module):
    """
    The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """

    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=z_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class TD_VAE(nn.Module):
    """
    The full TD_VAE model.

    -------------------
    First, let's first go through some definitions which would help understanding what is going
    on in the following code.
    Belief: As the model is fed a sequence of observations, x_t, the model updates its belief
        state, b_t, through an LSTM network. It is a deterministic function of x_t.

    -------------------
    First, let's first go through some definitions which would help understanding what is going
    on in the following code.

    Belief: As the model is fed a sequence of observations, x_t, the model updates its belief
        state, b_t, through an LSTM network. It is a deterministic function of x_t.
    State: The latent state variable, z.
    Observation: The observed variable, x.

    -------------------
    The TD_VAE contains several interconnected layers, which all have different purposes.
    We provide more details on each layer below, but, in summary, the layers are as follows:


    -------------------
    The TD_VAE contains several interconnected layers, which all have different purposes.
    We provide more details on each layer below, but, in summary, the layers are as follows:

    1) The preprocessing layer. This converts the input images to a form ready for LSTM training
    2) The LSTM network. This network computes the belief states of the preprocessed data X
    3) The encoder network. This is a two layer network that learns a compressed representation of
        the belief states. This compressed representation is called z, and the network is called P_B.
    4) The smoothing network. This is a two layer network that learns how to imagine/hallucinate/sample
        a compressed representation z from a previous time.
    5) The transition network. Also called state prediction network or forward model. This is a two layer
        network that enables us to predict compressed representation z in the future
    6) The decoder network. This layer will learn how to reverse the compression process and convert z
        back to the original input dimensions X

    -------------------
    The TD-VAE model learns by minimizing reconstruction loss (reconstructing compressed belief state
    back to original X dimensions) and mimizing two distinct KL divergence terms. To describe these terms,
    it is helpful to realize that there are three ways to arrive at a compressed representation z.

    A) Encoder network (P_B)
    B) Smoothing network (q_s)
    C) Transition network (P_T)

    The first KL term used to learn is: KL( P_B | q_s )
    This learns if the world is providing enough information to predict an earlier compressed state.

    The second KL term used to learn is: KL( P_B | P_T ), which is learned via sampling (no closed solution)
    This learns if the future compressed states can be predicted from past information.

    Therefore, the objective function for the TD-VAE is as follows:

            loss = ||X - decoder(P_B(X))|| + KL( P_B | q_s ) + KL( P_B | P_T )

    -------------------
    We are primarily interested in two aspects of this model.

    1) We would like to be able to predict future states of the world, perhaps from static images.
    2) We would like to explore the encoded representation z formed from the belief state encoder.
    We will compare z across different contexts, which will tell us how different contexts impact
    behavior in the world over time.
    """

    def __init__(
        self,
        x_size,
        processed_x_size,
        b_size,
        z_size,
        d_block_hidden_size,
        decoder_hidden_size,
    ):
        super(TD_VAE, self).__init__()
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size
        self.d_block_hidden_size = d_block_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        ##############################
        ############ Initialize layers
        ##############################
        # 1) Pre-process layer
        self.process_x = PreProcess(
            input_size=self.x_size, processed_x_size=self.processed_x_size
        )

        # 2) One layer LSTM for aggregating belief states
        self.lstm = nn.LSTM(
            input_size=self.processed_x_size, hidden_size=self.b_size, batch_first=True
        )

        ##############################
        ######### Layers in Appendix D
        ##############################
        # Note: Weights shared across time but not across layers.

        #########
        # Layers 3 and 4 accomplish sampling of compressed state z.
        # This is learning the belief distribution (P_B)
        #########

        # 3) Second layer DBlock compressing belief state
        # e.g. For compressing belief at time 2 (b_t2) to infer z at time 2 from layer 2
        # This layer is used in line 1 of Appendix D
        # Input: the belief state
        # Output: the layer 2 encoded representation (compressed representation) z
        # Remember, b is output from the LSTM
        self.encoder_b_to_z_layer2 = DBlock(
            input_size=self.b_size,
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        # 4) First layer DBlock compressing belief state
        # This layer is used in line 2 of Appendix D
        # Input: the belief state and also the output from the previous layer (encoder_b_to_z_layer2)
        # Output: the full encoded representation (compressed representation) z
        self.encoder_b_to_z_layer1 = DBlock(
            input_size=self.b_size + self.z_size,
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        #########
        # Layers 5 and 6 sample (or imagine/hallucinate) a compressed state z at a previous time (e.g. time 1).
        # The sampling uses information about the prior belief state (e.g. time 1) and the current
        # compressed state z (e.g. time 2). In other words, for example, given the compressed state at
        # time 2 (z_t2) and computed belief at time 1 (b_t1), sample a compressed state at time 1 (z_t1).
        #
        # These layers learn the smoothing distribution (q_s), which we use as a second way of computing a
        # compressed representation z, but at an earlier time point (e.g. time 1). This can be considered an
        # "ex post guess of the state of the world", or, indirectly examining how much the world can be
        # predicted. Later, when we calculate the loss (see calculate_loss), this smoothing distribution is
        # learned by trying to mimic the encoder distribution, and can therefore be viewed as a pseudo cycle-loss.
        #########

        # 5) Second layer DBlock compressing imagined z at time 2 and belief state at time 1.
        # Taking as input b_t1 and z_t2 (where z_t2 = [z_t2_layer1, z_t2_layer2])
        # This is line 5 of Appendix D.
        # Because z_t2 is two layers concatenated, multiply z_size by 2 below
        # Input: the belief state at past time and compressed state at current time
        # Output: the layer 2 encoded representation (compressed representation) z (alternate calculation to encoding)
        self.smooth_previous_z_layer2 = DBlock(
            input_size=(self.b_size + (2 * self.z_size)),
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        # 6) First layer DBlock compressing imaged z at time 2, belief state at time 1, and
        # compressed state z at time 1 from layer 1. In other words, this layer takes as input:
        # b_t1, z_t2, and the previous layer z_t1_layer2 (z_time1_layer2_dblock)
        # This is line 7 of Appendix D.
        # Because z_t2 is two layers concatenated, multiply z_size by 2 below, also
        # add a z_size because z_t1_layer1 needs info from z_t2_layer2
        # Input: the belief state at past time, compressed state at current time, and smoothing layer 1
        # Output: the full encoded representation (compressed representation) z (alternate calculation to encoding)
        self.smooth_previous_z_layer1 = DBlock(
            input_size=(self.b_size + (2 * self.z_size) + self.z_size),
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        #########
        # Layers 7 and 8 represent the forward pass of learning how the world evolves.
        # Given the compressed state of the world imagined at a previous time from the smoothing distribution,
        # predict how the state will change at time 2.
        #
        # These layers learn the transition distribution (P_T). This nework enables us to make predictions, and in
        # a "jumpy" fashion (which means far into the future, not just step by step)
        #########

        # 7) Second layer DBlock compressing imagined state z at time 1.
        # This represents line 9 in Appendix D
        # Because z_time1 is two layers of z, multiply z_size by 2 below
        # Input: compressed representation z
        # Output: Layer 2 prediction of future state z
        self.transition_z_layer2 = DBlock(
            input_size=2 * self.z_size,
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        # 8) First layer DBlock compressing imagined state z at time 1. For sampling how world evolves
        # This represents line 10 in Appendix D
        # Because z_time1 is two layers of z, multiply z_size by 2 below and add z_size to incoporate
        # prevous layer
        # Input: compressed representation z, and transition network layer 2
        # Output: Full prediction of future state z
        self.transition_z_layer1 = DBlock(
            input_size=(2 * self.z_size) + self.z_size,
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        #########
        # Layer 9 represents the decoder, which decodes the compressed state into the observed state.
        #
        # This layer enables us to learn through a reconstruction loss and also will fully enable us to
        # sample future predictions.
        #########

        # 9) Decoder from state z to observation x
        # This represents line 11 in Appendix D
        # Because z is two layers, multiply z_size by 2 below
        # Input: Compressed representation z
        # Output: X_hat (estimated X) of the original input data dimensions (per frame only)
        self.decoder_z_to_x = Decoder(
            z_size=2 * self.z_size,
            hidden_size=self.decoder_hidden_size,
            x_size=self.x_size,
        )

    def forward(self, images, t):
        self.batch_size = images.size()[0]
        self.x = images

        # Pre-precess image x
        self.processed_x = self.process_x(self.x)

        # Aggregate the belief b - the model only uses b in subsequent steps
        self.b, (self.h_n, self.c_n) = self.lstm(self.processed_x)

        """
        Calculate the VD-VAE loss, which corresponds to equations (6) and (8) in the TD-VAE paper.

        We provide more explicit details below, but, in summary, the loss contains three core elements:
        1) The difference between the encoded z at time 1 and the hallucinated (or imagined) z at time 1

        We provide more explicit details below, but, in summary, the loss contains three core elements:

        1) The difference between the encoded z at time 1 and the hallucinated (or imagined) z at time 1

            The encoded z at time 1 is generated two ways: 1) the belief distribution (P_B) and 2) the
            smoothing distribution (q_s). The smoothing distribution learns the compressed z at time 2 from
            information of compressed z at time 1 and belief state at time 2. This loss term ensures that
            the hallucination of future data looks like how future data might be expected to behave given
            certain states of the world that were previously observed. Importantly, we can sample from the
            belief distribution to generate new data.
                This term enables data generation.

        2) The difference between the encoded z at time 2 and the predicted z at time 2

                This term enables data generation.

        2) The difference between the encoded z at time 2 and the predicted z at time 2

            The encoded z at time 2 is generated two ways as well: 1) the belief distribution (P_B), which
            is the same way we encode z at time 1 above, and 2) the state prediction network (or forward model).
            This loss term ensures that we're able to predict the compressed state of the world in a jumpy manner
            (meaning, more than one step in the future) and that this prediction is similar to our encoder model.
            Importantly, we can make predictions about future states with the transition network.
                This term enables state prediction.
        3) The reconstruction term between the decoded z at time 2 and real data at time 2

                This term enables state prediction.

        3) The reconstruction term between the decoded z at time 2 and real data at time 2

            This is a standard reconstruction term. We pass input data x through an encoder (belief network) to
            a lower dimensional compressed z. We then decode this z through a decoder to obtain the same input
            dimensions as X. We minimize the difference between this decoded X and the ground truth X. Importantly,
            with this decoder we can now fully simulate new data in the original dimension.

                This term fully enables data generation (original dimension).
        """

        ############################
        # Prior to calculating the loss, sample from the previously defined layers.
        # We will use these layers in subsequent components of the loss function.
        ############################
        # 1) compress z at time 2 using information from belief state at time 2.
        (
            z_layer2_mu,
            z_layer2_logsigma,
        ) = self.encoder_b_to_z_layer2(self.b[:, t, :])

        z_layer2_epsilon = torch.randn_like(
            z_layer2_mu
        )  # Note, this is the reparameterization

        z_layer2 = z_layer2_mu + torch.exp(z_layer2_logsigma) * z_layer2_epsilon

        # Sample z at time 2 from the belief state at time 2 (layer 1)
        (
            z_layer1_mu,
            z_layer1_logsigma,
        ) = self.encoder_b_to_z_layer1(torch.cat((self.b[:, t, :], z_layer2), dim=-1))

        z_layer1 = z_layer1_mu + torch.exp(z_layer1_logsigma) * z_layer2_epsilon

        # Concatenate z from layer 1 and layer 2
        z = torch.cat((z_layer1, z_layer2), dim=-1)

        return z
