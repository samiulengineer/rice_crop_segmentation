import os
import cv2
import math
import json
import rasterio
import matplotlib
import numpy as np
import pandas as pd
from config import *
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
import traceback 

# from utils import video_to_frame
matplotlib.use("Agg")



def pct_clip(array, pct=[2.5, 97.5]):
    array_min, array_max = np.nanpercentile(array, pct[0]), np.nanpercentile(array, pct[1])
    clip = (array - array_min) / (array_max - array_min)
    clip[clip > 1] = 1
    clip[clip < 0] = 0
    return clip


def transform_data(label, num_classes):
    """
    Summary:
        transform label/mask into one hot matrix and return
    Arguments:
        label (arr): label/mask
        num_classes (int): number of class in label/mask
    Return:
        one hot label matrix
    """
    # return the label as one hot encoded
    return to_categorical(label, num_classes)


def read_img(directory, in_channels=None, label=False, patch_idx=None, height=256, width=256):
    """
    Summary:
        read image with rasterio and normalize the feature
    Arguments:
        directory (str): image path to read
        in_channels (bool): number of channels to read
        label (bool): TRUE if the given directory is mask directory otherwise False
        patch_idx (list): patch indices to read
    Return:
        numpy.array
    """

    # for musk images
    if label:
        with rasterio.open(directory) as fmask: # opening the directory
            mask = fmask.read(1)    # read the image (Data from a raster band can be accessed by the band’s index number. Following the GDAL convention, bands are indexed from 1. [int or list, optional] – If indexes is a list, the result is a 3D array, but is a 2D array if it is a band index number.
        
        mask[mask == 2.0] = 0
        mask[mask == 1.0] = 1

        mask = mask.astype("int32")
    
        if patch_idx:
            # extract patch from original mask
            return mask[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]]
        else:
            return mask #np.expand_dims(mask, axis=2)
    # for features images
    else:
        # read N number of channels
        with rasterio.open(directory) as inp:
            X =inp.read()
        X= np.swapaxes(X,0,2)
        for i in range(3):
            X[i,:,:]= pct_clip(X[i,:,:])
        X = (X - mean) / std
        if patch_idx:
            # extract patch from original features
            return X[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3], :]
        else:
            return X


def data_split(images, masks):
    """
    Summary:
        split dataset into train, valid and test
    Arguments:
        images (list): all image directory list
        masks (list): all mask directory
    Return:
        return the split data.
    """
    # splitting training data
    x_train, x_rem, y_train, y_rem = train_test_split(
        images, masks, train_size=train_size, random_state=42
    )
    # splitting test and validation data
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_rem, y_rem, test_size=test_size, random_state=42
    )
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def save_csv(dictionary, name):
    """
    Summary:
        save csv file
    Arguments:
        dictionary (dict): data as a dictionary object
        name (str): file name to save
    Return:
        save file
    """
    # check for target directory
    if not os.path.exists(dataset_dir / "data/csv"):
        try:
            os.makedirs(dataset_dir / "data/csv")  # making target directory
        except Exception as e:
            print(e)
            raise
    # converting dictionary to pandas dataframe
    df = pd.DataFrame.from_dict(dictionary)
    # from dataframe to csv
    df.to_csv((dataset_dir / "data/csv" / name), index=False, header=True)


def video_to_frame():
    """
    Summary:
        create frames from video
    Arguments:
        empty
    Return:
        frames
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(
            str(root_dir / "data/video_frame" / "frame_%06d.jpg" % count), image
        )  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


def data_csv_gen():
    """
    Summary:
        splitting data into train, test, valid
    Arguments:
        empty
    Return:
        save file
    """
    images = []
    image_names = os.listdir(image_path)
    image_names = sorted(image_names)
    for i in image_names:
        images.append(image_path / i)
        
    if evaluation:
        # creating dictionary for evaluation
        eval = {"feature_ids": images, "masks": images}
        # saving dictionary as csv files
        save_csv(eval, "eval.csv")
    else:
        masks = []
        mask_names = os.listdir(mask_path)
        mask_names = sorted(mask_names)
        for i in mask_names:
            masks.append(mask_path / i)

        x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(images, masks)

        # creating dictionary for train, test and validation
        train = {"feature_ids": x_train, "masks": y_train}
        valid = {"feature_ids": x_valid, "masks": y_valid}
        test = {"feature_ids": x_test, "masks": y_test}

        # saving dictionary as csv files
        save_csv(train, "train.csv")
        save_csv(valid, "valid.csv")
        save_csv(test, "test.csv")
        

def class_percentage_check(label):
    """
    Summary:
        check class percentage of a single mask image
    Arguments:
        label (numpy.ndarray): mask image array
    Return:
        dict object holding percentage of each class
    """
    # calculating total pixels
    total_pix = label.shape[0] * label.shape[0]
    # get the total number of pixel labeled as 1
    class_one = np.sum(label)
    # get the total number of pixel labeled as 0
    class_zero_p = total_pix - class_one
    # return the pixel percent of each class
    return {
        "zero_class": ((class_zero_p / total_pix) * 100),
        "one_class": ((class_one / total_pix) * 100),
    }


def save_patch_idx(path, patch_size=patch_size, stride=stride, test=None, patch_class_balance=None):
    """
    Summary:
        finding patch image indices for single image based on class percentage. work like convolutional layer
    Arguments:
        path (str): image path
        patch_size (int): size of the patch image
        stride (int): how many stride to take for each patch image
    Return:
        list holding all the patch image indices for a image
    """
    with rasterio.open(path) as t:  # opening the image directory
        img = t.read(1)
    img[img == 2] = 0 # convert unlabeled to non-water/background

    # calculating number patch for given image
    patch_height = int((img.shape[0] - patch_size) / stride) + 1
    patch_weight = int((img.shape[1] - patch_size) / stride) + 1

    # total patch images = patch_height * patch_weight
    patch_idx = []

    # image column traverse
    for i in range(patch_height + 1):
        # get the start and end row index
        s_row = i * stride
        e_row = s_row + patch_size
        
        if e_row > img.shape[0]:
            s_row = img.shape[0] - patch_size
            e_row = img.shape[0]
        
        if e_row <= img.shape[0]:
            # image row traverse
            for j in range(patch_weight + 1):
                # get the start and end column index
                start = j * stride
                end = start + patch_size
                
                if end > img.shape[1]:
                    start = img.shape[1] - patch_size
                    end = img.shape[1]
                
                if end <= img.shape[1]:
                    tmp = img[s_row:e_row, start:end]  # slicing the image
                    percen = class_percentage_check(tmp)  # find class percentage

                    # take all patch for test images
                    if not patch_class_balance or test == 'test':
                        patch_idx.append([s_row, e_row, start, end])

                    # store patch image indices based on class percentage
                    else:
                        if percen["one_class"] > class_balance_threshold:
                            patch_idx.append([s_row, e_row, start, end])
                            
                if end == img.shape[1]:
                    break
            
        if e_row == img.shape[0]:
            break  
            
    return patch_idx


def write_json(json_path, json_file, data):
    """
    Summary:
        save dict object into json file
    Arguments:
        target_path (str): path to save json file
        target_file (str): file name to save
        data (dict): dictionary object holding data
    Returns:
        save json file
    """
    # check for target directory
    if not os.path.exists(json_path):
        try:
            os.makedirs(json_path)  # making target directory
        except Exception as e:
            print(e)
            raise
    # writing the json file
    with open(json_file, "w") as f:
        json.dump(data, f)


def patch_images(data, name):
    """
    Summary:
        save all patch indices of all images
    Arguments:
        data: data file contain image paths
        name (str): file name to save patch indices
    Returns:
        save patch indices into file
    """
    img_dirs = []
    masks_dirs = []
    all_patch = []

    # loop through all images
    for i in range(len(data)):
        # fetching patch indices
        patches = save_patch_idx(
            data.masks.values[i],
            patch_size=patch_size,
            stride=stride,
            test=name.split("_")[0],
            patch_class_balance=patch_class_balance,
        )

        # generate data point for each patch image
        for patch in patches:
            img_dirs.append(data.feature_ids.values[i])
            masks_dirs.append(data.masks.values[i])
            all_patch.append(patch)
    # dictionary for patch images
    temp = {"feature_ids": img_dirs, "masks": masks_dirs, "patch_idx": all_patch}

    # save data to json 
    write_json(json_dir, json_dir / f"{name}_p_{patch_size}_s_{stride}.json", temp)


# Data Augment class
# ----------------------------------------------------------------------------------------------
class Augment:
    def __init__(self, batch_size, channels, ratio=0.3, seed=42):
        super().__init__()
        """
        Summary:
            Augmentaion class for doing data augmentation on feature images and corresponding masks
        Arguments:
            batch_size (int): how many data to pass in a single step
            ratio (float): percentage of augment data in a single batch
            seed (int): both use the same seed, so they'll make the same random changes.
        Return:
            class object
        """
        self.ratio = ratio
        self.channels = channels
        self.aug_img_batch = math.ceil(batch_size * ratio)
        self.aug = A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Blur(p=0.5),
            ]
        )

    def call(self, feature_dir, label_dir, patch_idx=None):
        """
        Summary:
            randomly select a directory and augment data
            from that specific image and mask
        Arguments:
            feature_dir (list): all train image directory list
            label_dir (list): all train mask directory list
        Return:
            augmented image and mask
        """
        # choose random number from given limit
        aug_idx = np.random.randint(0, len(feature_dir), self.aug_img_batch)
        features = []
        labels = []
        # get the augmented features and masks
        for i in aug_idx:
            # get the patch image and mask
            if patch_idx:
                img = read_img(
                    feature_dir[i], in_channels=self.channels, patch_idx=patch_idx[i]
                )
                mask = read_img(label_dir[i], label=True, patch_idx=patch_idx[i])

            else:
                # get the image and mask
                img = read_img(feature_dir[i], in_channels=self.channels)
                mask = read_img(label_dir[i], label=True)

            # augment the image and mask
            augmented = self.aug(image=img, mask=mask)
            features.append(augmented["image"])
            labels.append(augmented["mask"])
        return features, labels


# Dataloader class
# ----------------------------------------------------------------------------------------------
class MyDataset(Sequence):
    def __init__(
        self,
        img_dir,
        tgt_dir,
        in_channels,
        batch_size,
        num_class,
        patchify,
        transform_fn=None,
        augment=None,
        weights=None,
        patch_idx=None,
    ):
        """
        Summary:
             MyDataset class for creating dataloader object
        Arguments:
            img_dir (list): all image directory
            tgt_dir (list): all mask/ label directory
            in_channels (int): number of input channels
            batch_size (int): how many data to pass in a single step
            patchify (bool): set TRUE if patchify experiment
            transform_fn (function): function to transform mask images for training
            num_class (int): number of class in mask image
            augment (object): Augment class object
            weight (list): class weight for imbalance class
            patch_idx (list): list of patch indices
        Return:
            class object
        """
        self.img_dir = img_dir
        self.tgt_dir = tgt_dir
        self.patch_idx = patch_idx
        self.patchify = patchify
        self.in_channels = in_channels
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.num_class = num_class
        self.augment = augment
        self.weights = weights

    def __len__(self):
        """
        return total number of batch to travel full dataset
        """
        # getting the length of batches
        return len(self.img_dir) // self.batch_size

    def __getitem__(self, idx):
        """
        Summary:
            create a single batch for training
        Arguments:
            idx (int): sequential batch number
        Return:
            images and masks as numpy array for a single batch
        """
        # get index for single batch
        batch_x = self.img_dir[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.tgt_dir[idx * self.batch_size : (idx + 1) * self.batch_size]

        # get patch index for single batch
        if self.patchify:
            batch_patch = self.patch_idx[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]

        imgs = []
        tgts = []
        # get all image and target for single batch
        for i in range(len(batch_x)):
            if self.patchify:
                # get image from the directory
                imgs.append(
                    read_img(
                        batch_x[i],
                        in_channels=self.in_channels,
                        patch_idx=batch_patch[i],
                    )
                )
                # transform mask for model (categorically)
                if self.transform_fn:
                    tgts.append(
                        self.transform_fn(
                            read_img(batch_y[i], label=True, patch_idx=batch_patch[i]),
                            self.num_class,
                        )
                    )
                # get the mask without transform
                else:
                    tgts.append(
                        read_img(batch_y[i], label=True, patch_idx=batch_patch[i])
                    )
            else:
                imgs.append(read_img(batch_x[i], in_channels=self.in_channels))
                # transform mask for model (categorically)
                if self.transform_fn:
                    tgts.append(
                        self.transform_fn(
                            read_img(batch_y[i], label=True), self.num_class
                        )
                    )
                # get the mask without transform
                else:
                    tgts.append(read_img(batch_y[i], label=True))

        # augment data using Augment class above if augment is true
        if self.augment:
            if self.patchify:
                aug_imgs, aug_masks = self.augment.call(
                    self.img_dir, self.tgt_dir, self.patch_idx
                )  # augment patch images and mask randomly
                imgs = imgs + aug_imgs  # adding augmented images

            else:
                aug_imgs, aug_masks = self.augment.call(
                    self.img_dir, self.tgt_dir
                )  # augment images and mask randomly
                imgs = imgs + aug_imgs  # adding augmented images

            # transform mask for model (categorically)
            if self.transform_fn:
                for i in range(len(aug_masks)):
                    tgts.append(self.transform_fn(aug_masks[i], self.num_class))
            else:
                tgts = tgts + aug_masks  # adding augmented masks

        # converting list to numpy array
        tgts = np.array(tgts)
        imgs = np.array(imgs)

        # return weighted features and labels
        if self.weights is not None:
            # creating a constant tensor
            class_weights = tf.constant(self.weights)
            class_weights = class_weights / tf.reduce_sum(class_weights)  # normalizing the weights
            # get the weighted target
            y_weights = tf.gather(class_weights, indices=tf.cast(tgts, tf.int32))  
            return tf.convert_to_tensor(imgs), y_weights

        # return tensor that is converted from numpy array
        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts)

    def get_random_data(self, idx=-1):
        """
        Summary:
            randomly chose an image and mask or the given index image and mask
        Arguments:
            idx (int): specific image index default -1 for random
        Return:
            image and mask as numpy array
        """
        
        if idx != -1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.img_dir))

        imgs = []
        tgts = []
        if self.patchify:
            imgs.append(read_img(
                self.img_dir[idx], in_channels=self.in_channels, patch_idx=self.patch_idx[idx]))

            # transform mask for model
            if self.transform_fn:
                tgts.append(
                    self.transform_fn(
                        read_img(
                            self.tgt_dir[idx], label=True, patch_idx=self.patch_idx[idx]
                        ),
                        self.num_class,
                    )
                )
            else:
                tgts.append(
                    read_img(
                        self.tgt_dir[idx], label=True, patch_idx=self.patch_idx[idx]
                    )
                )
        else:
            imgs.append(read_img(self.img_dir[idx], in_channels=self.in_channels))

            # transform mask for model
            if self.transform_fn:
                tgts.append(
                    self.transform_fn(
                        read_img(self.tgt_dir[idx], label=True), self.num_class
                    )
                )
            else:
                tgts.append(read_img(self.tgt_dir[idx], label=True))

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts), idx


def data_train_val_test_dataloader(mode='train'): #dataloader
    """
    Load train, validation, or test datasets based on the specified mode.
    
    Args:
        mode (str): 'train', 'val', or 'test'
    
    Returns:
        Dataset object for the specified mode
    """
    global train_dir, valid_dir, eval_dir, p_train_dir, p_valid_dir, p_eval_dir, test_dir, p_test_dir
    
    # Determine appropriate directories and patch naming based on mode
    if mode == 'train':
        var_list = [train_dir, p_train_dir]
        patch_name = "train_phr_cb" if patch_class_balance else "train_phr"
        print("(Patchify = {}) Loading Train features and masks directories.....".format(patchify))
    elif mode == 'val':
        var_list = [valid_dir, p_valid_dir]
        patch_name = "valid_phr_cb" if patch_class_balance else "valid_phr"
        print("(Patchify = {}) Loading Validation features and masks directories.....".format(patchify))
    else:  # 'test' or 'eval'
        var_list = [eval_dir, p_eval_dir] if evaluation else [test_dir, p_test_dir]
        patch_name = "eval_phr_cb" if evaluation else "test_phr_cb"
        print("(Patchify = {}) Loading Test/Evaluation features and masks directories.....".format(patchify))
    
    # Generate CSV if not exists
    if not os.path.exists(var_list[0]):
        data_csv_gen()
    
    # Create patchify indices if not exists and patchify is True
    if not os.path.exists(var_list[1]) and patchify:
        print(f"Saving patchify indices for {mode}....")
        data = pd.read_csv(var_list[0])
        patch_images(data, patch_name)
    
    # Load features and masks
    if patchify:
        with var_list[1].open() as j:
            dir_data = json.loads(j.read())
        features = dir_data["feature_ids"]
        masks = dir_data["masks"]
        patch_idx = dir_data["patch_idx"]
    else:
        dir_data = pd.read_csv(var_list[0])
        features = dir_data.feature_ids.values
        masks = dir_data.masks.values
        patch_idx = None
    
    #print(f"{mode.capitalize()} Examples: {}".format(len(features)))
    
    # Create augmentation object for training
    if mode == 'train' and augment and batch_size > 1:
        augment_obj = Augment(batch_size, in_channels)
        n_batch_size = batch_size - augment_obj.aug_img_batch
        weights = tf.constant(balance_weights) if weights else None
    else:
        augment_obj = None
        n_batch_size = batch_size
        weights = None
    
    # Create dataset
    dataset = MyDataset(
        features,
        masks,
        in_channels=in_channels,
        patchify=patchify,
        batch_size=n_batch_size if mode == 'train' else batch_size,
        transform_fn=transform_data,
        num_class=num_classes,
        augment=augment_obj if mode == 'train' else None,
        weights=weights,
        patch_idx=patch_idx,
    )
    
    return dataset