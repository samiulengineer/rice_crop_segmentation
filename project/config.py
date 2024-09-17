from datetime import date
from pathlib import Path

# Default configuration values
config = {
    "dtype": "nsr-2",
    "mean":"0.1495",
    "std":"0.0825",
    "rename": False, # this parameter is not required, need to check
    "root_dir": Path("/mnt/hdd2/mdsamiul/project/rice_crop_segmentation"),
    
    # Image In/Out Parameters
    # -------------------------
    "channel_type": "rgb",
    "in_channels": 3,
    "num_classes": 3,
    "height": 400,
    "width": 400,
    
    # Training Parameters
    # -------------------------
    "model_name": "planet-2",
    "batch_size": 1,
    "epochs": 3000,
    "learning_rate": 3e-4,
    "val_plot_epoch": 2,
    "augment": False,
    "transfer_lr": False,
    "gpu": "0",
    
    # Class Balance Parameters
    # -------------------------
    "weights": False,
    "balance_weights": [2.2, 7.8, 0],
    "patchify": True,
    "patch_class_balance": True,
    "train_size": 0.8,
    "patch_size": 2048,
    "stride": 1024,
    
    # Log Parameters
    # -------------------------
    "csv": True,
    "val_pred_plot": True,
    "lr": True,
    "tensorboard": True,
    "early_stop": False,
    "checkpoint": True,
    "patience": 300, # required for early_stopping, if accuracy does not change for "patience" epochs, model will stop automatically
    
    # Evaluation Parameters
    # -------------------------
    "load_model_name": 'planet-2_ex_2024-04-29_e_3000_p_2048_s_1024_nsr-1_ep_3000not.hdf5',
    "load_model_dir": None, #  If None, then by befault root_dir/model/model_name/load_model_name
    "evaluation": False,
    "video_path": None,  # If None, then by default root_dir/data/video_frame
    "index": -1, # this parameter is not required, need to check
}

# Update mean and std based on dtype
def update_dtype_config(dtype):
    dtype_config = {
        "nsr-1": (0.1495, 0.0825),
        "nsr-2": (0.1836, 0.1195),
        "nsr-3": (0.152, 0.0876),
        "nsr-4": (0.0356, 0.0824),
        "nsr-5": (0.0306, 0.0747),
        "sar-1": (-11.6673, 5.1528),
        "sar-2": (-10.9157, 4.8009),
        "sar-3": (-10.066135, 4.200516),
        "sar-4": (0.75226486, 2.6914093),
        "sar-5": (1.6018738, 2.857205)
    }
    return dtype_config.get(dtype, (None, None))

# Update in_channels based on channel_type
def update_in_channels(channel_type):
    return 3 if channel_type == "rgb" else len(channel_type)

# Update config based on args
def update_config(args):
    for key, value in vars(args).items():
        print(f"KEY:{key} Value: {value}")
        if value is not None:
            config[key] = value
    
    # Ensure root_dir is a Path object
    if isinstance(config['root_dir'], str):
        config['root_dir'] = Path(config['root_dir'])
    
    config['mean'], config['std'] = update_dtype_config(config['dtype'])
    
    if config['mean'] is None or config['std'] is None:
        print(f"Error: dtype '{config['dtype']}' is not recognized.")
    
    config['in_channels'] = update_in_channels(config['channel_type'])
    
    # Update dependent paths
    config['dataset_dir'] = config['root_dir'] / f"data/{config['dtype']}"
    config['train_dir'] = config['dataset_dir'] / "data/csv/train.csv"
    config['valid_dir'] = config['dataset_dir'] / "data/csv/valid.csv"
    config['test_dir'] = config['dataset_dir'] / "data/csv/test.csv"
    config['eval_dir'] = config['dataset_dir'] / "data/csv/eval.csv"
    config['p_train_dir'] = config['dataset_dir'] / f"data/json/train_patch_phr_cb_{config['patch_size']}_{config['stride']}.json"
    config['p_valid_dir'] = config['dataset_dir'] / f"data/json/valid_patch_phr_cb_{config['patch_size']}_{config['stride']}.json"
    config['p_test_dir'] = config['dataset_dir'] / f"data/json/test_patch_phr_cb_{config['patch_size']}_{config['stride']}.json"
    config['p_eval_dir'] = config['dataset_dir'] / f"data/json/eval_patch_phr_cb_{config['patch_size']}_{config['stride']}.json"
    config['experiment'] = f"{str(date.today())}_e_{config['epochs']}_p_{config['patch_size']}_s_{config['stride']}_{config['dtype']}"
    config['tensorboard_log_name'] = "{}_ex_{}".format(config['model_name'], config['experiment'])
    config['tensorboard_log_dir'] = config['root_dir'] / "logs/tens_logger" / config['model_name']
    config['csv_log_name'] = "{}_ex_{}.csv".format(config['model_name'], config['experiment'])
    config['csv_log_dir'] = config['root_dir'] / "logs/csv_logger" / config['model_name']
    config['csv_logger_path'] = config['root_dir'] / "logs/csv_logger"
    config['checkpoint_name'] = "{}_ex_{}.hdf5".format(config['model_name'], config['experiment'])
    config['checkpoint_dir'] = config['root_dir'] / "logs/model" / config['model_name']
    if config['load_model_dir'] is None:
        config['load_model_dir'] = config['root_dir'] / "logs/model" / config['model_name']
    config['prediction_test_dir'] = config['root_dir'] / "logs/prediction" / config['model_name'] / "test" / config['experiment']
    config['prediction_eval_dir'] = config['root_dir'] / "logs/prediction" / config['model_name'] / "eval" / config['experiment']
    config['prediction_val_dir'] = config['root_dir'] / "logs/prediction" / config['model_name'] / "validation" / config['experiment']
    config['visualization_dir'] = config['root_dir'] / "logs/visualization"

    # Adjust height and width if patchify is True
    if config['patchify']:
        config['height'] = config['patch_size']
        config['width'] = config['patch_size']
