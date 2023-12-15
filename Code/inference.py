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
            
def make_predictions(test_dataset=test_batches, output_dir='../Result', num=5):
    os.makedirs(output_dir, exist_ok=True)

    for i, (image, mask) in enumerate(test_dataset.take(num)):
        pred_mask = model.predict(image)
        pred_mask = create_mask(pred_mask)

        # Create output folders
        save_path = os.path.join(output_dir, f"prediction_{i+1}")
        os.makedirs(save_path, exist_ok=True)

        # Save the images to the output directory
        plt.imshow(image[0].numpy())
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f"input_image_{i+1}.png"))
        plt.close()
        
        # Save the mask using plt
        plt.imshow(mask[0].numpy().squeeze())  # assuming mask is grayscale
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f"label_mask_{i+1}.png"))
        plt.close()

        # Save the predicted mask using plt
        plt.imshow(pred_mask.numpy().squeeze())  # assuming pred_mask is grayscale
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f"predicted_mask_{i+1}.png"))
        plt.close()
    
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(sample_image=sample_image, sample_mask=sample_mask)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
