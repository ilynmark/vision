# Contains the model inference code for processing a single image.
# It should take a single image path as input and save the output on Result folder.

import tensorflow as tf
import os
from IPython.display import clear_output
from model import *
from dataset import *

output_classes=3
model = unet_model(output_channels=output_classes)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_batches, test_batches, info, sample_image, sample_mask = get_dataset()

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1, sample_image=sample_image, sample_mask=sample_mask):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])
            
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(sample_image=sample_image, sample_mask=sample_mask)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
