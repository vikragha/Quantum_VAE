import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import csv

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

import time
import datetime
import pickle

with open('qm9_5k.npy', 'rb') as f:
    reduced_data = np.load(f)

reduced_data = np.array([[i.numpy() if i else 1e-10 for i in r] for r in reduced_data])
train_samples = reduced_data.shape[0]
x_train = reduced_data[:int(train_samples*0.85)]
x_test = reduced_data[int(train_samples*0.85):]

n_features = x_train.shape[1]

latent_dim = 56
batch_size = 32
patches_e = 8
patches_d = 8
quantum_e = True
quantum_d = False

n_single_features = n_features // patches_e

if quantum_e and quantum_d:
    MODEL_SAVE_DIR = "model/sq-vae/cifar%s-%s" % (patches_e, patches_d)
elif quantum_e and not quantum_d:
    MODEL_SAVE_DIR = "model/sq-vae-e/cifar%s" % patches_e
elif not quantum_e and quantum_d:
    MODEL_SAVE_DIR = "model/sq-vae-d/cifar%s" % patches_d
else:
    MODEL_SAVE_DIR = "model/vae/cifar"
    
MODEL_NAME = "latent-%s" % latent_dim

model_spec_name = "%s-model" % MODEL_NAME
model_rslt_name = "%s-results.pickle" % MODEL_NAME

model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model_ckpt_path = os.path.join(model_save_path, "learned-model")
model_spec_path = os.path.join(model_save_path, model_spec_name)
model_rslt_path = os.path.join(model_save_path, model_rslt_name)

n_qubits = int(math.log(n_single_features, 2))
# qml.enable_tape()
dev = qml.device("default.qubit.tf", wires=n_qubits)
@qml.qnode(dev, interface='tf', diff_method='backprop')
def qnode_e(inputs, weights):
    qml.templates.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize = True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

n_qubits_d = latent_dim // patches_d
dev_d = qml.device("default.qubit.tf", wires=n_qubits_d)
@qml.qnode(dev_d, interface='tf', diff_method='backprop')
def qnode_d(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits_d))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits_d))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_d)]

weight_shapes_e = {"weights": (3, n_qubits, 3)}
weight_shapes_d = {"weights": (3, n_qubits_d, 3)}

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_var = layers.Dense(latent_dim)
        self.final_layer_d = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(n_features)
        ])
        if quantum_e:
            self.qlayers_e = []
            for i in range(patches_e):
                self.qlayers_e.append(qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=n_qubits))

        if quantum_d:
            self.qlayers_d = []
            for i in range(patches_d):
                self.qlayers_d.append(qml.qnn.KerasLayer(qnode_d, weight_shapes_d, output_dim=n_qubits_d))
                
        self.encoder = tf.keras.Sequential([
              layers.Dense(512, activation='relu'),
              layers.Dense(256, activation='relu'),
              layers.Dense(64)
            ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(n_features)
        ])
    
    def encode(self, x):
        if quantum_e:
            split_x = tf.split(x, num_or_size_splits=patches_e, axis=-1)
            for i in range(patches_e):
                patch_x = self.qlayers_e[i](split_x[i])
                if i == 0:
                    result = patch_x
                else:
                    result = tf.concat([result, patch_x], -1)
        else:
            result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z):
        if quantum_d:
            split_z = tf.split(z, num_or_size_splits=patches_d, axis=-1)
            for i in range(patches_d):
                patch_z = self.qlayers_d[i](split_z[i])
                if i == 0:
                    result = patch_z
                else:
                    result = tf.concat([result, patch_z], -1)
            x = self.final_layer_d(result)
        else:
            x = self.decoder(z)
        return x
        
    def reparameterize(self, mu, logvar):
        std = tf.math.exp(0.5 * logvar)
        eps = tf.random.normal(std.shape)
        return eps * std + mu
    
    def call(self, x):
        [mu, log_var] = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)

        return [decoded, x, mu, log_var]
    
    def loss_function(self, *args):
        recons = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = batch_size / train_samples
        recons_loss =tf.reduce_mean((x - recons)**2, axis=-1, keepdims=True)
        kld_loss = tf.math.reduce_mean(-0.5 * tf.math.reduce_sum(1 + log_var - mu ** 2 - 
                                tf.math.exp(log_var), axis=-1, keepdims=True), axis=-1, keepdims=True)
        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    
    def sample(self, noise):
        samples = self.decode(noise)
        return samples
    
model = Autoencoder(latent_dim)

print('Start training...')

recons_losses = []
losses = []
fakes = []
times = []

start_time = time.time()
EPOCHS = 10
for epoch in range(EPOCHS):    
    batch_losses = []
    batch_recons_losses = []
    noise = np.random.normal(size=[6, latent_dim]).astype(np.float32)
    fakes.append(model.sample(noise))
    
    epoch_time = time.time()
    batches = len(x_train) // batch_size
    for batch in range(batches):
        x = x_train[batch_size * batch:min(batch_size * (batch + 1), len(x_train))]
        
        with tf.GradientTape() as t1, tf.GradientTape() as t2, tf.GradientTape() as t3, tf.GradientTape() as t4:
            results =  model(tf.cast(x, tf.float32))
            loss = tf.reduce_mean(model.loss_function(*results)['loss'])
            batch_losses.append(loss)
            batch_recons_losses.append(tf.reduce_mean(model.loss_function(*results)['Reconstruction_Loss']))

            if quantum_e:
                quantum_e_trainable_variables = model.qlayers_e[0].trainable_variables
                for i in range(1, patches_e):
                    quantum_e_trainable_variables += model.qlayers_e[i].trainable_variables
                
                grad_enc = t1.gradient(loss, 
                                       quantum_e_trainable_variables)
                model.optimizer.apply_gradients(zip(grad_enc,
                                                   quantum_e_trainable_variables))
            else:
                grad_enc = t1.gradient(loss, model.encoder.trainable_variables)
                model.optimizer.apply_gradients(zip(grad_enc, model.encoder.trainable_variables))

            grad_z = t2.gradient(loss, model.fc_mu.trainable_variables+model.fc_var.trainable_variables)
            model.optimizer.apply_gradients(zip(grad_z, model.fc_mu.trainable_variables + 
                                                model.fc_var.trainable_variables))
            
            if quantum_d:
                quantum_d_trainable_variables = model.qlayers_d[0].trainable_variables
                for i in range(1, patches_d):
                    quantum_d_trainable_variables += model.qlayers_d[i].trainable_variables
                
                grad_dec = t3.gradient(loss,
                                       quantum_d_trainable_variables
                                      )
                model.optimizer.apply_gradients(zip(grad_dec,
                                                   quantum_d_trainable_variables
                                                   ))
                grad_final = t4.gradient(loss, model.final_layer_d.trainable_variables)
                model.optimizer.apply_gradients(zip(grad_final, model.final_layer_d.trainable_variables))
                
            else:
                grad_dec = t3.gradient(loss, model.decoder.trainable_variables)
                model.optimizer.apply_gradients(zip(grad_dec, model.decoder.trainable_variables))

        print('Epoch {} Batch {}/{}\tLoss {:.4f}'.format(epoch+1, batch, batches, loss.numpy()), end='\r')
    
    epoch_loss = np.mean(batch_losses)
    losses.append(epoch_loss)
    epoch_recons_loss = np.mean(batch_recons_losses)
    recons_losses.append(epoch_recons_loss)
    epoch_t = time.time() - epoch_time
    times.append(epoch_t)
    
    # DEBUG -> Run test samples.
    with tf.GradientTape() as t1:
        results = model(tf.cast(x_test, tf.float32))
        test_loss = tf.reduce_mean(model.loss_function(*results)['loss'])
        
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print('Elapsed {}\t Epoch {}/{} \tTrain Loss {:.4f} Test Loss: {:.4f}'.format(et, \
                                                                epoch+1, EPOCHS, epoch_loss, test_loss))
    with open(model_rslt_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, epoch_loss, epoch_recons_loss, epoch_t])
        
    model.save_weights(model_ckpt_path + str(epoch))
