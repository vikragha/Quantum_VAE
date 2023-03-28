from collections import defaultdict
import pennylane as qml
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import argparse
import os
import math
import datetime
import time
from frechetdist import frdist
import rdkit
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
from util_dir.utils_io import random_string
from models import Generator, Discriminator, EncoderVAE
from data.sparse_molecular_dataset import SparseMolecularDataset
from rdkit import Chem
from pysmiles import read_smiles


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config, log=None):
        """Initialize configurations."""

        # Log
        self.log = log

        # Model configurations.
        self.z_dim = config.qubits
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.f_dim = self.data.features
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lambda_wgan = config.lambda_wgan
        self.lambda_rec = config.lambda_rec
        self.post_method = config.post_method
        
        self.metric = 'validity,qed'

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = (len(self.data) // self.batch_size)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout_rate = config.dropout
        self.n_critic = config.n_critic
        self.resume_epoch = config.resume_epoch
        
        # Training or testing.
        self.mode = config.mode

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device)

        # Directories.
        self.log_dir_path = config.log_dir_path
        self.model_dir_path = config.model_dir_path
        self.img_dir_path = config.img_dir_path

        # Step size.
        self.model_save_step = config.model_save_step

        # VAE KL weight.
        self.kl_la = 1.
        
        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)
        
        
        n_single_features = n_features // patches_e
        config.qubits = int(math.log(n_single_features, 2))
        qml.enable_tape()
        dev = qml.device("default.qubit.tf", wires=n_qubits)
        @qml.qnode(dev, interface='tf', diff_method='backprop')
        def qnode_e(inputs, weights):
          qml.templates.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize = True)
          qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
          return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
