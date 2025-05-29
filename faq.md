## **Data Visualization**
---

### **1. How can I plot results during the validation phase?**

During the training process, validation occurs at configurable intervals, determined by the `val_plot_epoch` variable in the `config.py` file. For example, if `val_plot_epoch` is set to 3, validation will be performed after every 3 epochs. 

The generated validation plots are automatically saved in the following directory:

```
root/logs/prediction/<model_name>/validation/<experiment_name>.jpg
```

Plotting can be further customized to allow **selective plotting** or **random plotting**. The process to achieve this is detailed in **Question 16**.

This setup ensures systematic tracking and flexible visualization of model performance during validation.

### 2. How can I visualize predictions during the testing phase?

To visualize predictions during the testing phase, follow these steps:

1. **Configure the Settings**: In the `config.py` file, ensure the following setting:
    ```python
    evaluation = False
    ```
   This ensures predictions are generated on the test dataset without performing evaluation.

2. **Execute the Command**: Run the testing script.

3. **Access the Plots**: The resulting prediction plots will be saved in the following directory:
    ```
    root_dir/logs/prediction/model_name/test/experiment
    ```
   - Replace `root_dir` with the root directory of your project.
   - Replace `model_name` with the name of your model.
   - Replace `experiment` with the specific testing experiment name.

This setup allows you to generate and store visualized predictions for the test dataset.


### 3. How can I visualize predictions during the evaluation phase?

To visualize predictions during the evaluation phase, follow these steps:

1. **Execute the Evaluation Script**: Run the `test.py` script and ensure that `evaluation = True` is set in the `config.py` file.

2. **Generate Predictions**: The script will generate predictions using the evaluation dataset.

3. **View Resulting Plots**: The resulting prediction plots will be automatically saved in the following folder structure:

```
root_dir/logs/prediction/model_name/eval/experiment
```

- Replace `root_dir` with the root directory of your project.
- Replace `model_name` with the name of your model.
- Replace `experiment` with the specific experiment or evaluation run.

This allows you to visually assess the model's performance during evaluation.

### **4. How can I visualize the dataset?**

Visualizing the dataset before training is a common practice in deep learning. To visualize the dataset, use the `display_all` function in the `visualization.ipynb` file.

#### **Example:**
```python
display_all(
    data=train_df,
    name="train",
    visualization_dir=visualization_dir
)
```

*For test dataset, pass `data=test_df` with `name="test"`, and similarly for validation.*

#### **Set the Visualization Directory:**
```python
visualization_dir = pathlib.Path("path_where_you_want_to_store_plots")
```

This function generates and saves visualizations, enabling you to inspect your dataset effectively before training.

## 5. How can I randomly or selectively plot data samples during training?

Plotting data samples during training helps verify preprocessing, augmentations, and dataset integrity. The behavior is controlled by the `index` variable in `config.py`:
```python
index = "random"  # For random plotting
```
- Set `index = 1` to plot the first validation sample consistently.
- Set `index = "random"` to plot samples randomly during training.

The plots can be found in the directory:
```
root/logs/prediction/model_name/validation/experiment_name.jpg
```

## Dataset Preprocessing


### 6. What is the significance of the CSV and JSON files, and what information do they typically contain?

CSV and JSON files organize the dataset for training, testing, and evaluation. The CSV file contains the directories for the training, testing, and validation images where each CSV file contains **`feature_ids`** and **`masks`**, where **`feature_ids`** represent the original images and **`masks`** represent the ground truth of the data. It serves as a reference for locating the dataset split during the training process, while JSON files store patch metadata, including source image details and patch coordinates.

The JSON file contains details about patches such as their coordinates and corresponding source images. For example:

```json
{
  "feature_ids": ["image1.jpg", "image1.jpg"],
  "masks": ["image1_mask.jpg", "image1_mask.jpg"],
  "patch_idx": [[0, 0, 512, 512], [512, 0, 1024, 512]]
}
```
These files ensure proper data handling during training and preprocessing, avoiding errors due to inconsistencies.


### **7. Why is it important to check for unique image heights and widths before training?**

It is crucial to ensure that all images have the same shape before training to maintain consistency during preprocessing and avoid errors in patch creation. 

During preprocessing, images are divided into smaller patches based on a specified patch size. If the patch size does not align correctly with the image dimensions, it can cause errors or inconsistent model performance. Ensuring uniform image dimensions and setting an appropriate patch size is essential for stable and effective training.

To check the unique image heights and widths:
1. Use the `check_height_width(data_dir)` function in the `visualization.py` file. This function will identify unique image dimensions in the dataset. 
2. If the shapes vary throughout the dataset, set `patch_size = min(shape)` to ensure that the code runs smoothly. 
3. Without this adjustment, mismatched shapes could pass into the model, leading to processing errors.

Properly aligning image dimensions and patch size ensures the pipeline operates seamlessly.

### **8. How can I adjust class weights to address class imbalance issues?**

To handle class imbalance, class weights can be adjusted during training to ensure the model gives appropriate importance to underrepresented classes. Here's how:

#### **Binary Classification**
- For imbalanced datasets (e.g., one class has significantly more samples), enable class weighting by setting `weights = True` in the `config.py` file.
- Define appropriate values for `balance_weights`. For example:
  ```python
  balance_weights = [4, 6]  # Class 0 gets a weight of 4, Class 1 gets a weight of 6
  ```
  These weights should be proportional to the class distributions.

#### **Multi-class Classification**
- In cases where one class should be ignored (e.g., a boundary class), set its weight to `0`. For example:
  ```python
  balance_weights = [4, 6, 0]  # Class 2 (boundary) is ignored
  ```

This approach adjusts the loss function to account for class proportions, improving performance on underrepresented classes while ignoring irrelevant ones.

#### **Calculating Class Weights**
You can compute class weights using the `class_balance_check` function in the `visualization.ipynb` file:
1. Run the function with your training dataset:
   ```python
   class_balance_check(patchify=False, data_dir=train_df)
   ```
2. Example output:
   ```
   class pixel: 1.0 = 27.747506680695906
   class pixel: 2.0 = 72.2524933193041
   ```
   Here, 27.75% of pixels belong to Class 1 and 72.25% to Class 2.

3. Assign weights inversely proportional to these percentages. For instance:
   - Weight for Class 1: \( 72.25 / 10 ~ 7.2 \)
   - Weight for Class 2: \( 27.75 / 10 ~ 2.7 \)
   ```python
   balance_weights = [7.2, 2.7]
   ```
   
Update the `config.py` file as follows:
```python
weights = True
balance_weights = [7.2, 2.7]
```

This ensures the model emphasizes underrepresented classes during training, mitigating imbalance issues.
### 9. Why is it necessary to delete CSV and JSON files before starting a new training session on modified data?

In this pipeline, if a CSV file already exists, the system will not generate a new one, and the same applies to JSON files. Therefore, if you make any changes to the `config.py` file that could affect the training data (e.g., data paths, preprocessing steps, or class distributions), it is necessary to manually delete the existing CSV and JSON files. This ensures that the changes are reflected in the newly generated training dataset.

You can find the CSV and JSON files in the following directories:

```
root/data/csv
```
```
root/data/json
```

Deleting these files before starting a new training session ensures that the training dataset is updated according to the latest configuration.
### **10. How can I extract the mean and standard deviation from the `visualization.py` file, and why are these metrics important?**

Normalization using mean and standard deviation ensures consistent input distributions, which stabilizes training and accelerates convergence. To extract the mean and standard deviation of your dataset, use the `calculate_average` function in the `visualization.ipynb` file. Follow these steps:

#### **Steps to Calculate Mean and Standard Deviation**
1. Prepare the input feature paths:
   ```python
   features_path = train_df["feature_ids"].to_list()
   ```
2. Compute the mean and standard deviation by running:
   ```python
   mean, std_dev = calculate_average(features_path)
   ```

3. Configure the computed mean and standard deviation in the `config.py` file. Use the dictionary format for datasets with different distributions:

```python
mean_std = {
    "nir": [0.1495, 0.0825],
    "red": [0.1836, 0.1195],
    "swir": [0.152, 0.0876],
    "vh+swir": [0.0356, 0.0824],
    "vh": [-11.6673, 5.1528],
    "vv": [-10.9157, 4.8009],
    "vh-vv": [-10.066135, 4.200516],
    "vv+swir": [0.75226486, 2.6914093]
}

mean = mean_std.get(dir_name)[0]  # Mean value based on `dir_name`
std = mean_std.get(dir_name)[1]   # Standard deviation based on `dir_name`
```
### **11. How do I check the number of classes in the dataset?**

To determine the number of classes in your dataset, use the `class_balance_check` function in the `visualization.ipynb` file. This function provides a breakdown of the unique classes along with their distribution.

#### **Steps to Check the Number of Classes**
1. Load the training dataset:
   ```python
   train_df = pd.read_csv(train_dir)
   ```
2. Execute the `class_balance_check` function:
   ```python
   class_balance_check(patchify=False, data_dir=train_df)
   ```

#### **Example Output**
```plaintext
Class percentage:
class pixel: 1.0 = 27.747506680695906
class pixel: 2.0 = 72.2524933193041
Unique value in the mask dict_keys([1.0, 2.0])
```
In this example:
- The dataset contains **2 unique classes** (1.0 and 2.0).
- Update the `config.py` file accordingly:
  ```python
  num_classes = 2
  ``` 

This ensures that the model is correctly configured for the number of classes in your dataset.

### **12. How can I incorporate large-scale images into the pipeline?**

Large-scale images must be tiled to ensure they fit within memory constraints and allow efficient processing during training. For large-scale images, you can use the `save_tiles` function in the `visualization.ipynb` file. This function splits large images into smaller tiles, making them manageable for training and visualization.

#### **Function Syntax**
```python
save_tiles(path, out_path, tiles_size=2048, stride=1024)
```

#### **Parameters**
- `path`: The directory containing the large images.
- `out_path`: The directory where the smaller tiles will be saved.
- `tiles_size`: The size of each tile (e.g., `2048x2048` pixels).
- `stride`: The overlap between consecutive tiles (e.g., `1024` pixels).

#### **Example**
```python
path = "data/large_images/"          # Directory with large images
out_path = "data/tiles/"             # Directory to save the tiles
tiles_size = 2048                    # Each tile will be 2048x2048 pixels
stride = 1024                        # Tiles will overlap by 1024 pixels

save_tiles(path, out_path, tiles_size, stride)
```
  
Using `save_tiles` allows efficient handling of large images by dividing them into manageable tiles, enabling seamless training and visualization.

### 13. How do I check the number of patches created during data preprocessing?

The number of patches created during preprocessing is determined using the formula:

<!-- <img src="https://latex.codecogs.com/svg.image?$$\text{number\_of\_patches}=\left\lceil\frac{\text{height}-\text{patch\_size}}{\text{stride}}&plus;1\right\rceil\times\left\lceil\frac{\text{width}-\text{patch\_size}}{\text{stride}}&plus;1\right\rceil$$" title="$$\text{number\_of\_patches}=\left\lceil\frac{\text{height}-\text{patch\_size}}{\text{stride}}+1\right\rceil\times\left\lceil\frac{\text{width}-\text{patch\_size}}{\text{stride}}+1\right\rceil$$" /> -->

<img src="https://latex.codecogs.com/svg.image?\inline&space;\bg{black}\;\text{Number&space;of&space;Patches}=\frac{(\text{height}&plus;\text{stride}-\text{patch&space;size})\cdot(\text{width}&plus;\text{stride}-\text{patch&space;size})}{\text{stride}^2}\;" title="\;\text{Number of Patches}=\frac{(\text{height}+\text{stride}-\text{patch size})\cdot(\text{width}+\text{stride}-\text{patch size})}{\text{stride}^2}\;" />

These patches are further processed and evaluated against the class balance threshold if `patch_class_balance` is set to `True` in the `config.py` file. All generated patches and their corresponding metadata are saved in the JSON file.

To count the patches, use the `number_of_patches()` function available in the `visualization.ipynb` file. This function reads the JSON file and calculates the total number of patches based on the metadata. Proper evaluation of patches ensures alignment with training requirements and dataset consistency.

### 14. How can I control the data split for training, validation, and testing?

Controlling the data split allows you to allocate specific proportions of the dataset for training, validation, and testing, ensuring proper evaluation and model performance.

In `config.py`, adjust the following variables:
```python
train_size = 0.8  # 80% of the data for training
test_size = 0.5   # 10% for testing and 10% for validation (remaining after training)
```
This ensures data is divided as required, with the remaining portion allocated for validation and testing.


## Model Training and Augmentation


### **15. What happens if `transfer_lr` is set to `False` and `load_model_name` is specified?**

To enable transfer learning, `transfer_lr` must be set to `True`. Otherwise, the pipeline will proceed with **fine-tuning** if a pre-trained model is specified.

If `transfer_lr = False` and a valid `load_model_name` is provided in the `config.py` file, the model will undergo **fine-tuning**. This means:

- The model will load the pre-trained weights from the specified file or directory in `load_model_name`.
- All layers of the model will be trainable, allowing the training process to update the weights across the entire network.
- Fine-tuning is particularly useful when adapting a pre-trained model to a similar task or dataset where additional training is required to optimize performance.

**Example:**
```python
dir_name = "vh-vv"  # Dataset name
model_name = "unet"  # Model name
load_model_name = "unet_ex_2024-12-03_e_2_p_2048_s_1024_vh-vv.keras"  # Load model name must match the actual file
```

*For training from scratch, under no circumstances should `load_model_name` be present in the `load_model_dir`. Otherwise, the pipeline will use the pre-trained model.*

If `load_model_name` is not provided, the model will be trained from scratch.

### 16. What is the process for adding data augmentation to the pipeline?
  
Data augmentation is a technique used to artificially expand the training dataset by applying transformations to the images. It helps improve model generalization by exposing it to varied versions of the same data, thereby reducing overfitting and improving robustness.

To enable data augmentation in the pipeline, follow these steps:

1. Set `augment = True` in the `config.py` file.
2. The pipeline currently includes the following augmentation techniques:
   - **VerticalFlip**: Flips the image vertically with a certain probability.
   - **HorizontalFlip**: Flips the image horizontally with a certain probability.
   - **RandomRotate90**: Rotates the image by 90 degrees randomly.
   - **Blur**: Applies a blur effect to the image.

3. To add more augmentation techniques, you can modify the `Augment` class located in the `dataset.py` file. Incorporate additional augmentation methods as needed to enhance the variability of your dataset.

**Example:**
```python
self.aug = A.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.5),
        A.CLAHE(p=0.5),  # Contrast Limited Adaptive Histogram Equalization
        A.ColorJitter(p=0.5),  # Random changes in brightness, contrast, saturation, and hue
    ]
)
```

By enabling and customizing augmentations, you can adapt the pipeline to specific requirements, improving the diversity and quality of training data for better model performance.

### 17. What is class balance threshold and how can I control it during training?

The class balance threshold determines the minimum percentage of the positive class that must be present in a patch image for it to be included in the training process. Any patch with a positive class proportion lower than the threshold will be discarded and not used for training.

To adjust the class balance threshold, modify the `class_balance_threshold` variable in the `config.py` file based on your requirements. This provides flexibility in controlling the inclusion criteria for patch images during training.

```python
# Example configuration in config.py
class_balance_threshold = 20  # 20% positive class required in a patch
```

- If `class_balance_threshold = 20`, only patches where the positive class (e.g., foreground pixels) covers **at least 20%** of the patch will be used for training.  
- If a patch has a positive class proportion of 15%, it will be discarded.
- If a patch has a positive class proportion of 25%, it will be included in the training set.

This mechanism ensures that patches with insufficient positive samples are excluded, improving the training process by focusing on meaningful patches.




### 18. What steps are required to add additional performance metrics to the training process?

Adding performance metrics allows for a more comprehensive evaluation of the model. Metrics like **Mean IoU** and **Dice Coefficient** are defined in the `metrics.py` file. To add new metrics, modify the `get_metrics()` function in `metrics.py`:
```python
def get_metrics():
    return {
        "MyMeanIoU": MyMeanIoU(num_classes),
        "f1-score": tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.9),
        "dice_coef_score": dice_coef_score,
    }
```

These additional metrics will be calculated during training and validation, providing deeper insights into model performance.


### 19. How can I configure the `config.py` file step by step for my dataset using the `visualization.ipynb` file?

To configure the `config.py` file for your dataset, follow these systematic steps:

#### **Dataset Preparation**

Ensure the dataset directory is correctly structured as follows:

```
data/
└── dataset_type_or_name/   # Use a short and descriptive name
    ├── groundtruth/        # Directory for ground truth masks
    ├── input/              # Directory for input images
```

After confirming the structure, proceed with the steps below:

#### **Step 1: Generate Train, Test, and Validation Datasets**

1. Open the `visualization.ipynb` file.
2. Use the `data_csv_gen()` function to create CSV files for the train, test, and validation splits.
   ```python
   data_csv_gen()
   ```
3. These CSV files will contain paths to the input images and their corresponding masks, as explained in **Question 6**.

#### **Step 2: Patch Images and Create JSON Files**

1. Use the `patch_images()` function to patch the dataset and generate a JSON file containing patch indices.
   ```python
   patch_images(train_df, "train_phr_cb")
   ```
   - `train_df`: The training dataset DataFrame created in Step 1.
   - `"train_phr_cb"`: The naming convention for the generated JSON file.

2. Ensure patching parameters (`patch_size`, `stride`, `class_balance_threshold`) are correctly configured in the `config.py` file, as detailed in **Question 13**, **Question 17**.


#### **Step 3: Configure Class Balance Weights**

1. If class balance weights are required, enable them in the `config.py` file:
   ```python
   weights = True
   ```
2. Use the `class_balance_check()` function to:
   - Determine the unique classes in your dataset.
   - Calculate appropriate weights for each class:
     ```python
     class_balance_check(patchify=True, data_dir=train_df)
     ```
3. Update the `balance_weights` and `num_classes` variables in the `config.py` file based on the output, as explained in **Question 8**.

#### **Step 4: Validate Image Dimensions**

1. Run the `check_height_width()` function to inspect the dimensions of the dataset.
   ```python
   check_height_width(data_dir)
   ```
   - This step ensures that all images are consistent in height and width, and identifies any anomalies.

2. Use this information to adjust the `patch_size`, `height`, and `width` variables in the `config.py` file, as explained in **Question 7**.

#### **Step 5: Calculate Mean and Standard Deviation**

1. Use the `calculate_average()` function to compute the mean and standard deviation of your dataset:
   ```python
   mean, std_dev = calculate_average(features_path)
   ```
   - `features_path`: List of feature image paths from the dataset.

2. Update the `mean_std` dictionary in the `config.py` file with the computed values:
   ```python
   mean_std = {
       "dataset_name": [mean, std_dev]
   }
   mean = mean_std.get(dir_name)[0]
   std = mean_std.get(dir_name)[1]
   ```


<!--### Summary of Configuration Steps

1. **Ensure dataset structure is correct.**
2. **Generate train, test, and validation datasets** using the `data_csv_gen()` function.
3. **Patch images and generate JSON files** using the `patch_images()` function.
4. **Enable class balance weights** and configure them using the `class_balance_check()` function.
5. **Validate image dimensions** and adjust patch size and resolution settings in `config.py`.
6. **Calculate and configure mean and standard deviation** for normalization. -->

By following these steps, your `config.py` file will be properly configured to handle your dataset efficiently, ensuring smooth execution of the pipeline.