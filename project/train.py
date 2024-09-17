import argparse
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--experiment", type=str)
parser.add_argument("--gpu", type=str)
parser.add_argument("--dtype", type=str)  # Add dtype argument

args = parser.parse_args()

# Update config with parsed arguments
print("\n-------------------------------------------------------------------")
print("Taking value via ArgParser")
print("-------------------------------------------------------------------\n")
update_config(args)



import os
import time
import numpy as np
from loss import *
from metrics import get_metrics
import tensorflow_addons as tfa
from dataset import get_train_val_dataloader
from tensorflow.keras.models import load_model
from model import get_model, get_model_transfer_lr
from utils import SelectCallbacks, create_paths
import random
import tensorflow as tf

# Parsing variable


# Ensure seed is set for reproducibility
seed_num = random.randint(1, 1000)
random.seed(seed_num)
tf.config.optimizer.set_jit("True")

print(f"root_dir:{config['root_dir']} model_name: {config['model_name']} epochs:{config['epochs']} batch_size:{config['batch_size']} experiment:{config['experiment']} gpu:{config['gpu']}")

# Set up train configuration
create_paths(test=False)

# setup GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Print Experimental Setup before Training
print("\n--------------------------------------------------------")
print("Model = {}".format(config['model_name']))
print("Epochs = {}".format(config['epochs']))
print("Batch Size = {}".format(config['batch_size']))
print("Preprocessed Data = {}".format(os.path.exists(config['train_dir'])))
print("Class Weight = {}".format(str(config['weights'])))
print("Experiment = {}".format(str(config['experiment'])))
print("--------------------------------------------------------\n")

# Dataset
train_dataset, val_dataset = get_train_val_dataloader()

# Metrics
metrics = list(get_metrics().values())
custom_obj = get_metrics()

# Optimizer
learning_rate = config['learning_rate']
weight_decay = 0.0001
adam = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay)

# Loss Function
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
custom_obj['loss'] = focal_loss()

# Compile
if (os.path.exists(config['load_model_dir'] / config['load_model_name'])) and config['transfer_lr']:
    print("\n---------------------------------------------------------------------")
    print("Transfer Learning from model checkpoint {}".format(config['load_model_name']))
    print("---------------------------------------------------------------------\n")
    model = load_model((config['load_model_dir'] / config['load_model_name']), custom_objects=custom_obj, compile=True)
    model = get_model_transfer_lr(model, config['num_classes'])
    model.compile(optimizer=adam, loss=loss, metrics=metrics)
else:
    if (os.path.exists(config['load_model_dir'] / config['load_model_name'])):
        print("\n---------------------------------------------------------------")
        print("Fine Tuning from model checkpoint {}".format(config['load_model_name']))
        print("---------------------------------------------------------------\n")
        model = load_model((config['load_model_dir'] / config['load_model_name']), custom_objects=custom_obj, compile=True)
    else:
        print("\n-------------------------------------------------------------------")
        print("Training only")
        print("-------------------------------------------------------------------\n")
        model = get_model()
        model.compile(optimizer=adam, loss=loss, metrics=metrics)

# Callbacks
loggers = SelectCallbacks(val_dataset, model)
model.summary()

# Check dataset shape
# If train_dataset is a generator, fetch one batch
images, labels = next(iter(train_dataset))
print(f"Image batch shape: {np.array(images).shape}")
print(f"Label batch shape: {np.array(labels).shape}")

# Check the input shape of the model
print("Model input shape:", model.input_shape)

# Fit
t0 = time.time()
# history = model.fit(train_dataset,
#                     verbose=1,
#                     epochs=config['epochs'],
#                     validation_data=val_dataset,
#                     shuffle=False,
#                     callbacks=loggers.get_callbacks(val_dataset, model),
#                     )

print("\n----------------------------------------------------")
print("training time minute: {}".format((time.time()-t0)/60))
print("\n----------------------------------------------------")
