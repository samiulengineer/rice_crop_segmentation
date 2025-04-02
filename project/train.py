import os
import time
import random
from loss import *
import numpy as np
from config import *
import tensorflow as tf
from metrics import get_metrics
from dataset import data_train_val_test_dataloader
from keras.models import load_model
from model import get_model, get_model_transfer_lr
from utils import SelectCallbacks, create_paths

# Ensure seed is set for reproducibility
seed_num = random.randint(1, 1000)
random.seed(seed_num)
tf.config.optimizer.set_jit("True")

# Set up train configuration
create_paths(test=False)

# setup GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Print Experimental Setup before Training
print("\n--------------------------------------------------------")
print("Model = {}".format(model_name))
print("Epochs = {}".format(epochs))
print("Batch Size = {}".format(batch_size))
print("Preprocessed Data = {}".format(os.path.exists(train_dir)))
print("Class Weight = {}".format(str(weights)))
print("Experiment = {}".format(str(experiment)))
print("dataset_dir = {}".format(str(dataset_dir)))
print("--------------------------------------------------------\n")

# Dataset
train_dataset= data_train_val_test_dataloader("train")
val_dataset= data_train_val_test_dataloader("val")

# Metrics
metrics = list(get_metrics().values())
custom_obj = get_metrics()

# Optimizer
learning_rate = learning_rate
weight_decay = 0.0001
adam = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, weight_decay=weight_decay)

# Loss Function
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
custom_obj['loss'] = focal_loss()

# Compile
if (os.path.exists(load_model_dir / load_model_name)) and transfer_lr:
    print("\n---------------------------------------------------------------------")
    print("Transfer Learning from model checkpoint {}".format(load_model_name))
    print("---------------------------------------------------------------------\n")
    model = load_model((load_model_dir / load_model_name), custom_objects=custom_obj, compile=True)
    model = get_model_transfer_lr(model, num_classes)
    model.compile(optimizer=adam, loss=loss, metrics=metrics)
else:
    if (os.path.exists(load_model_dir / load_model_name)):
        print("\n---------------------------------------------------------------")
        print("Fine Tuning from model checkpoint {}".format(load_model_name))
        print("---------------------------------------------------------------\n")
        model = load_model((load_model_dir / load_model_name), custom_objects=custom_obj, compile=True)
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

print("-------------------------------------------------------------------\n")
print(f"Image batch shape: {np.array(images).shape}")
print(f"Label batch shape: {np.array(labels).shape}")

# Check the input shape of the model
print("Model input shape:", model.input_shape)
print("-------------------------------------------------------------------\n")

# Fit
t0 = time.time()
history = model.fit(train_dataset,
                    verbose=1,
                    epochs=epochs,
                    validation_data=val_dataset,
                    shuffle=False,
                    callbacks=loggers.get_callbacks(val_dataset, model),
                    )

print("\n----------------------------------------------------")
print("training time minute: {}".format((time.time()-t0)/60))
print("\n----------------------------------------------------")