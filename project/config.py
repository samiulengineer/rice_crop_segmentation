from datetime import date
import pathlib

# rename = False # this parameter is not required, need to check

# Initial Directory
# -------------------------
dir_name = "vv"
root_dir = pathlib.Path("/mnt/hdd2/mdsamiul/project/rice_crop_segmentation")

# Image In/Out Parameters
# -------------------------
# "channel_type" = "rgb"
in_channels = 3
num_classes = 3
height = 512 # only required for cfr and cfr-cb, otherwise patch_size = height
width = 512  # only required for cfr and cfr-cb, otherwise patch_size = width

# Training Parameters
# -------------------------
model_name = "planet"
batch_size = 1
epochs = 4000
learning_rate = 3e-4
val_plot_epoch = 15
augment = True
transfer_lr = False
gpu = "0"

# Class Balance Parameters
# -------------------------
weights = True
balance_weights = [2.7, 7.3]
patchify = True
patch_class_balance = True
train_size = 0.8
patch_size = 2048
stride = 1024
if patchify:
    height = patch_size
    width = patch_size

# Log Parameters
# -------------------------
csv = True
val_pred_plot = True
lr = True
tensorboard = True
checkpoint = True
early_stop = False
patience = 300 # required for early_stopping, if accuracy does not change for "patience" epochs, model will stop automatically

# Experiment Name
# -------------------------
experiment = f"{str(date.today())}_e_{epochs}_p_{patch_size}_s_{stride}_{dir_name}"

# Evaluation Parameters
# -------------------------
load_model_name = 'planet-2_ex_2024-04-29_e_3000_p_2048_s_1024_nsr-1_ep_3000not.hdf5'
load_model_dir = root_dir / "logs/model" / model_name
evaluation = False
video_path = None  # If None then by default root_dir/data/video_frame
index = -1 # this parameter is not required, need to check
prediction_test_dir = root_dir / "logs/prediction" / model_name / "test" / experiment
prediction_eval_dir = root_dir / "logs/prediction" / model_name / "eval" / experiment
prediction_val_dir = root_dir / "logs/prediction" / model_name / "validation" / experiment

# CSV Directory
# -------------------------
dataset_dir = root_dir / f"data/signal_based_data/{dir_name}"
train_dir = dataset_dir / "data/csv/train.csv"
valid_dir = dataset_dir / "data/csv/valid.csv"
test_dir = dataset_dir / "data/csv/test.csv"
eval_dir = dataset_dir / "data/csv/eval.csv"

# Patchify Directory
# -------------------------
p_train_dir = dataset_dir / f"data/json/train_patch_phr_cb_{patch_size}_{stride}.json"
p_valid_dir = dataset_dir / f"data/json/valid_patch_phr_cb_{patch_size}_{stride}.json"
p_test_dir = dataset_dir / f"data/json/test_patch_phr_cb_{patch_size}_{stride}.json"
p_eval_dir = dataset_dir / f"data/json/eval_patch_phr_cb_{patch_size}_{stride}.json"

# Log Directory
# -------------------------
tensorboard_log_name = "{}_ex_{}".format(model_name, experiment)
tensorboard_log_dir = root_dir / "logs/tens_logger" / model_name
csv_log_name = "{}_ex_{}.csv".format(model_name, experiment)
csv_log_dir = root_dir / "logs/csv_logger" / model_name
csv_logger_path = root_dir / "logs/csv_logger"
checkpoint_name = "{}_ex_{}.hdf5".format(model_name, experiment)
checkpoint_dir = root_dir / "logs/model" / model_name

# Visualization Directory
# -------------------------
# visualization_dir = root_dir / "logs/visualization"

# Mean & Std
# -------------------------
mean_std = {
    "nir": [0.5001, 0.2852],
    "red": [0.3301, 0.2620],
    "swir": [0.152, 0.0876],
    "nsr-4": [0.0356, 0.0824],
    "nsr-5": [0.0306, 0.0747],
    "vh": [-11.6673, 5.1528],
    "vv": [-10.9157, 4.8009],
    "vh-vv": [-10.066135, 4.200516],
    "sar-4": [0.75226486, 2.6914093],
    "sar-5": [1.6018738, 2.857205]
}

mean = mean_std.get(dir_name)[0] # mean value will be based on dir_name
std = mean_std.get(dir_name)[1] # std value will be based on dir_name