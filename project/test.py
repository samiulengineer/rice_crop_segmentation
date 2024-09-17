import argparse
from config import config, update_config

# Parsing variable
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--model_name")
parser.add_argument("--load_model_name", type=str)
parser.add_argument("--index", type=int)
parser.add_argument("--experiment", type=str)
parser.add_argument("--gpu", type=str)
parser.add_argument("--evaluation", action='store_true')
parser.add_argument("--video_path", type=str)
args = parser.parse_args()

# Update config with parsed arguments
update_config(args)

import os
import time
from loss import *
from metrics import *
from tensorflow.keras.models import load_model
from dataset import get_test_dataloader
from utils import create_paths, test_eval_show_predictions, frame_to_video



print(f"root_dir:{config['root_dir']} model_name: {config['model_name']} load_model_name:{config['load_model_name']} index:{config['index']} experiment:{config['experiment']} gpu:{config['gpu']} evaluation:{config['evaluation']} video_path:{config['video_path']}")

# Training Start Time
t0 = time.time()

# Set up test configuration
if config['evaluation']:
    create_paths(eval=True)
    print("evaluation")
else:
    create_paths(test=True)
    print("test")

# setup GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Load Model
print("Loading model {} from {}".format(config['load_model_name'], config['load_model_dir']))
model = load_model((config['load_model_dir'] / config['load_model_name']), compile=False)

# Dataset
test_dataset = get_test_dataloader()

# Prediction Plot
print("--------------------------------------")
print("Saving test/evaluation predictions...")
print("--------------------------------------")
test_eval_show_predictions(test_dataset, model)
print("Call test_eval_show_predictions")
print("--------------------------------------")

# Test Score
if not config['evaluation']:
    metrics = list(get_metrics().values())
    adam = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=adam, loss=focal_loss(), metrics=metrics)
    model.evaluate(test_dataset)

# Frame to Video
if config['video_path'] == True:
    fname = config['dataset_dir'] + "prediction.avi"
    frame_to_video(fname, fps=30)

# Training time Calculation (End)
print("training time sec: {}".format((time.time() - t0)))

print("--------------------------------------")
print(f"saving prediction: {config['prediction_val_dir']}")
print("--------------------------------------")
