# Contains the training codes. It must accept a dataset path and hyperparameters as inputs. It should produce and save at least one checkpoint as output.

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from dataset import *
from inference import *

train_batches, test_batches, info, sample_image, sample_mask = get_dataset()

# Define a ModelCheckpoint callback
checkpoint_path = 'model_checkpoint.h5'
checkpoint_callback = ModelCheckpoint(checkpoint_path, 
                                      save_best_only=True,
                                      monitor='val_accuracy',
                                      mode='max',
                                      verbose=1)

def train(epochs=20, val_subsplits=5, batch_size=64):
    val_steps = info.splits['test'].num_examples//batch_size//val_subsplits
    
    train_length = info.splits['train'].num_examples
    steps_per_epoch = train_length // batch_size

    model_history = model.fit(train_batches, epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=val_steps,
                              validation_data=test_batches,
                              callbacks=[DisplayCallback(), checkpoint_callback])
    return model_history

def visualize_loss(model_history):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()