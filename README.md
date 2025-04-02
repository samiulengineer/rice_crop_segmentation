## **Introduction**

The pipeline offers a robust solution for handling image data of arbitrary sizes by employing a systematic approach of patching images into fixed-size patches. This not only enables efficient data augmentation but also facilitates training models on very high-resolution data that would otherwise exceed GPU memory limitations.

The process of training begins with the `train.py` script, where paths are created and essential utilities from `utils.py` are invoked. The data loading and preparation steps are handled in `dataset.py`, where the dataset is generated, split, and train, valid and test CSV files are saved. The image patching process, including saving co-ordinate of patches to train, valid, and json files is executed, followed by data augmentation and transformation through the `Augment` class. The dataset is then prepared using `MyDataset` class, which handles data fetching and transformation.

Metrics calculation, focal loss computation, and model initialization are handled in separate modules respectively `metrics.py`, `loss.py`, `model.py`, ensuring modularity and ease of management. Callback selection, including learning rate scheduling and validation visualization, is orchestrated through `SelectCallbacks` class.

For evaluation, two distinct scenario are provided in `test.py`. In the first scenario (evaluation = False), the evaluation dataset is prepared similarly to the training dataset, and predictions are made and displayed for analysis. In the second scenario (evaluation = True), additional steps for generating evaluation CSVs named eval_csv_gen is called

Throughout the pipeline, emphasis is placed on modularity, allowing for easy integration of new components or modifications. Understanding the intricacies of this pipeline is paramount for leveraging its capabilities effectively. For a deeper dive into its workings, referencing the FAPNET paper is recommended, providing insights into the architecture and rationale behind its design.


## **Dataset**

## Input Data Organization Guidelines

To maintain proper organization of input features, it is recommended to follow these guidelines:

### Separate Folders for Input Features and Masks

Ensure that input features and their respective masks are stored in separate folders. Use consistent naming conventions across these folders to facilitate sorting and alignment.

#### Example Naming Convention:

For instance, if the input feature is named `rice_01.png`, the corresponding mask should also be named `rice_01.png`.

Keep the above mention dataset in the data folder that give you following structure. Please do not change the directory name `image` and `gt_image`.

```
--data
    image
        rice_01.png
        rice_02.png
            ..
    gt_image
        rice_01.png
        rice_02.png
            ..
```



## **Model**

Our current pipeline supports semantic segmentation for both binary and multi-class tasks. To configure the pipeline for your desired number of classes, simply adjust the `num_classes` variable in the `config.py` file. No modifications are required in the `model.py` file. This straightforward configuration process allows for seamless adaptation to different classification requirements without the need for extensive code changes.
Our current pipeline supports semantic segmentation for both binary and multi-class tasks. To configure the pipeline for your desired number of classes, simply adjust the `num_classes` variable in the `config.py` file. No modifications are required in the `model.py` file. This straightforward configuration process allows for seamless adaptation to different classification requirements without the need for extensive code changes.

## **Setup**

First clone the github repo in your local or server machine by following:

```
git clone https://github.com/samiulengineer/rice_crop_segmentation.git
```

Change the working directory to project root directory. Use Conda/Pip to create a new environment and install dependency from `requirement.txt` file. The following command will install the packages according to the configuration file `requirement.txt`.

```
pip install -r requirements.txt
```


## **Experiment**

* ### **Patchify Half Resolution with Class Balance (PHR-CB)**:
In this experiment we take a threshold value (19%) of water class and remove the patch images for each chip that are less than threshold value. This specific experiment is taken from `https://github.com/samiulengineer/imseg_sar_csml`

```
python train.py
```

```
Check these variables from config.py carefully before starting to train the model

1. root_dir <root dir, where all files stored>
2. model_name <check accurate model name form model.py get_model()>
3. epochs <epochs>
4. batch_size <batch_size>
5. gpu <put gpu number, if you have more than 1 gpu, otherwise 0>
6. dir_name <data folder you want to get>
7. patchify <must be True>
8. patch_class_balance <must be true>
9. patch_size <an int value which will be the model input shape and slicing the given input shape>
```


* ### **Patchify Half Resolution with Class Balance Weight (PHR-CBW)**:

### Resolving Data Imbalance Issues

To address data imbalance problems, one can utilize the following method:

If encountering a scenario where there are only two classes, with the first class representing 60% of the dataset and the second class comprising 40%, the imbalance can be rectified by setting `weights= True` and specifying `balance_weights = [4,6]` in the config.py file.

However, in cases where there are three classes, and one of them is considered a boundary that should be disregarded, the weight of the corresponding class must be set to 0. For instance, in the command line, this would be denoted as `balance_weights = [4,6,0]` in the config.py file. This specific experiment is taken from `https://github.com/samiulengineer/imseg_sar_csml`

```
python train.py
```

```
Check these variables from config.py carefully before starting to train the model

1. root_dir <root dir, where all files stored>
2. model_name <check accurate model name form model.py get_model()>
3. epochs <epochs>
4. batch_size <batch_size>
5. gpu <put gpu number, if you have more than 1 gpu, otherwise 0>
6. dir_name <data folder you want to get>
7. patchify <must be True>
8. patch_class_balance <must be true>
9. patch_size <an int value which will be the model input shape and slicing the given input shape>
10. weights <if it is true, model will take consideration of balance_weights>
11. balance_weights <need to check percentage of different classes, then use it as a list>
```

## Transfer Learning, Fine-tuning, and Training from Scratch

In this project, the behavior of the model training process is determined by certain variables in the `config.py` file. Here's how they influence the training approach:

## Transfer Learning, Fine-tuning, and Training from Scratch

In this project, the behavior of the model training process is determined by certain variables in the `config.py` file. Here's how they influence the training approach:

### **Transfer Learning**

- If `transfer_lr` variable is set to `True` and a `load_model_name` is provided in `config.py`, the model will undergo transfer learning.


### **Fine Tuning**

- If `transfer_lr` is set to `False` but a `load_model_name` is provided in `config.py`, the model will undergo fine-tuning.


### **Training from Scratch**

- If `transfer_lr` is set to `False` and no `load_model_name` is provided in `config.py`, the model will be trained from scratch.



## Testing

* ### **PHR-CB and PHR-CBW Experiment**

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5
```


### **Evaluation from Image**

If you have the images without mask, we need to do data pre-preprocessing before passing to the model checkpoint. In that case, run the following command to evaluate the model without any mask.

1. You can check the prediction of test images inside the `logs > prediction > YOUR_MODELNAME > eval > experiment`.

```
python project/test.py \
    --dataset_dir YOUR_IMAGE_DIR/ \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --gpu YOUR_GPU_NUMBER \
    --evaluation True
```
##### Example:
```
python project/test.py \
    --dataset_dir /home/projects/imseg_sar/eavl_data/ \
    --model_name fapnet \
    --load_model_name my_model.hdf5 \
    --gpu 0 \
    --evaluation True
```
##### Example:
```
python project/test.py \
    --dataset_dir /home/projects/imseg_sar/eavl_data/ \
    --model_name fapnet \
    --load_model_name my_model.hdf5 \
    --gpu 0 \
    --evaluation True
```


## **Plot During Validation**
During the training process, validation will occur at specific epochs, which can be configured within the `config.py` module. Within `config.py`, there exists a variable named `val_plot_epoch`. If its value is set to 3, validation will be conducted after every 3 epochs. The resulting validation plots will be saved in the following folder.

```
root/logs/prediction/model_name/validation/experiment
```


## **Plot During Test**
### Running Test.py for Evaluation

To perform predictions on the test dataset and save plots, execute the following command in the terminal:

```
python test.py --evaluation False
```

This command will generate predictions on the test dataset and store plots in the following directory:

```
root_dir/logs/prediction/model_name/test/experiment
```
## **Plot During Evaluation**

To initiate evaluation on the test.py script, execute the following command in the terminal:

```
test.py --evaluation True
```

Upon execution, predictions will be generated using the evaluation dataset, and the resulting plots will be saved in the designated folder structure:

```
root_dir/logs/prediction/model_name/eval/experiment
```

To organize multiple datasets within the `data` folder, each dataset folder should contain two subfolders: `image` and `gt_image`.

For instance:

```
data/
│
├── dataset1/
│   ├── image/
│   └── gt_image/
│
├── dataset2/
│   ├── image/
│   └── gt_image/
│
└── dataset3/
    ├── image/
    └── gt_image/
```

This structure ensures clarity and consistency in managing datasets within the project.

Keep the above mention dataset in the data folder that give you following structure. Please do not change the directory name `image` and `gt_image`.





## **Plot During Validation**
During the training process, validation will occur at specific epochs, which can be configured within the `config.py` module. Within `config.py`, there exists a variable named `val_plot_epoch`. If its value is set to 3, validation will be conducted after every 3 epochs. The resulting validation plots will be saved in the following folder.

```
root/logs/prediction/model_name/validation/experiment
```


## **Plot During Test**
### Running Test.py for Evaluation

To perform predictions on the test dataset and save plots, execute the following command in the terminal:

```
python test.py --evaluation False
```

This command will generate predictions on the test dataset and store plots in the following directory:

```
root_dir/logs/prediction/model_name/test/experiment
```
## **Plot During Evaluation**

To initiate evaluation on the test.py script, execute the following command in the terminal:

```
test.py --evaluation True
```

Upon execution, predictions will be generated using the evaluation dataset, and the resulting plots will be saved in the designated folder structure:

```
root_dir/logs/prediction/model_name/eval/experiment
```




## **Visualization of the Dataset**

To visualize the dataset, we must execute the display_all function as defined in the sar_visualization.ipynb notebook. This function necessitates the CSV file that corresponds to the dataset we intend to visualize, along with a name parameter to establish a folder where the figure's visualization will be stored. For instance, calling 
```
display_all(data=train_df, name="train") 
```
accomplishes this task.
To visualize the dataset, we must execute the display_all function as defined in the sar_visualization.ipynb notebook. This function necessitates the CSV file that corresponds to the dataset we intend to visualize, along with a name parameter to establish a folder where the figure's visualization will be stored. For instance, calling 
```
display_all(data=train_df, name="train") 
```
accomplishes this task.


## **Action tree**


## Training Script: `train.py`
If we run `train.py` then the the following functions will be executed in the mentioned following order.

- `create_paths(test=False)` - [utils.py]
- `get_train_val_dataloader()`
  - `data_csv_gen()`
    - `data_split()`
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
     - `write_json()`
  - `Augment()` - Class
    - `call()`
      - `read_img()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `get_metrics()` - [metrics.py]
  - `MyMeanIOU()` - Class
- `focal_loss()` - [loss.py]
- `get_model_transfer_lr(model, num_classes)` - [model.py]
- `get_model()` - [model.py]
  - `models` - call all model functions
- `SelectCallbacks()` - Class - [utils.py]
  - `__init__()`
  - `lr_scheduler()`
  - `on_epoch_end()`
    - `val_show_predictions()`
      - `read_img()`
      - `transform_data()`
      - `display()`
  - `get_callbacks()`

---

## Test Script: `test.py`

#### If we run `test.py` with `evaluation = False`, then the the following functions will be executed in the mentioned flow.


- `create_paths()`
- `get_test_dataloader()`
  - `data_csv_gen()`
    - `data_split()`
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
    - `write_json()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `test_eval_show_predictions()`*
  - `read_img()`
  - `transform_data()`
  - `display()`
- `get_metrics()`

---

#### If we run `test.py` with `evaluation = True`, then the the following functions will be executed in the mentioned flow.

- `create_paths()`
- `get_test_dataloader()`
  - `eval_csv_gen()`*
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
    - `write_json()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `test_eval_show_predictions()`*
  - `read_img()`
  - `transform_data()`
  - `display_label()`*
  - `display()`
- `get_metrics()`

---