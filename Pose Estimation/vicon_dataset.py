import math
import pickle

import matplotlib
import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import struct
import cv2 as cv
import constants
import matplotlib.pyplot as plt
import json
import image_modifications
import random

import image_processing

# General global variables for reading data input
base_directory_for_images = '/home/ryan/Real_Data_F1Tenth/paper_set_august_2023'
picture_width = 640
picture_height = 480
BORDER_CONSTANT = 0.1
class CustomImageDataset(Dataset):
    def __init__(self):
        random.seed(20)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set File Count and Maps To Files
        self.possible_images = len(next(os.walk(base_directory_for_images))[1])
        self.total_images = 0
        self.image_map = {}
        self.image_cache = {}
        for index in range(self.possible_images):
            with open(base_directory_for_images + "/data" + str(index) + "/pose.pkl", 'rb') as f:
                item = pickle.load(f)
                min_x = item[2]
                max_x = item[3]
                min_y = item[4]
                max_y = item[5]
                min_x = min_x * 1 / picture_width
                max_x = max_x * 1 / picture_width
                min_y = min_y * 1 / picture_height
                max_y = max_y * 1 / picture_height
                # Check if car to close to side of car to be in camera view
                if abs(item[0][2]) < abs(item[0][0])/1.3:
                    continue
                if item[0][2] < 0.25:
                    continue
                if min_x < BORDER_CONSTANT and max_x > 1-BORDER_CONSTANT and min_y < BORDER_CONSTANT and max_y > 1-BORDER_CONSTANT:
                    continue
                if abs(float(min_x) - float(max_x)) <= 0.01 or abs(float(min_y) - float(max_y)) <= 0.01:
                    continue
                elif (max_y - min_y) / ((max_x - min_x) * 8) > 1:
                    continue
                else:
                    self.image_map[self.total_images] = index
                    self.total_images += 1

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        # print(idx)
        # if idx in self.image_cache:
        #     return self.image_cache[idx]
        image = torch.zeros(7, constants.pose_network_width, constants.pose_network_height, dtype=torch.float).to(self.device)
        label_before_tensor = [0, 0, 0, 0]
        # Get Input Image
        current_image_file_path = base_directory_for_images + "/data" + str(self.image_map[idx]) + "/img.png"
        file_image = None
        item = None
        if self.image_map[idx] in self.image_cache:
            file_image, item = self.image_cache[self.image_map[idx]]
        else:
            file_image = cv.imread(current_image_file_path, cv.IMREAD_COLOR)
            with open(base_directory_for_images + "/data" + str(self.image_map[idx]) + "/pose.pkl", 'rb') as f:
                item = pickle.load(f)
            self.image_cache[self.image_map[idx]] = file_image, item
        # file_image = cv.imread(current_image_file_path, cv.IMREAD_COLOR)
        unmodified_camera_image = cv.cvtColor(file_image, cv.COLOR_BGR2RGB)
        # unmodified_camera_image = image_modifications.change_brightness(unmodified_camera_image, value=(-80 + random.random() * 160))
        # unmodified_camera_image = image_modifications.noisy("gaussian", unmodified_camera_image)
        # with open(base_directory_for_images + "/data" + str(self.image_map[idx]) + "/pose.pkl", 'rb') as f:
        #     item = pickle.load(f)
        min_x = item[2]
        max_x = item[3]
        min_y = item[4]
        max_y = item[5]
        min_x = min_x * 1 / picture_width
        max_x = max_x * 1 / picture_width
        min_y = min_y * 1 / picture_height
        max_y = max_y * 1 / picture_height

        # RANDOMIZE BBOX SLIGHTLY
        percent_BBOX_randomization = 0.2
        min_x = min_x + (max_x-min_x) * (2*(0.5 - random.random())) * percent_BBOX_randomization
        max_x = max_x + (max_x - min_x) * (2 * (0.5 - random.random())) * percent_BBOX_randomization
        min_y = min_y + (max_y - min_y) * (2 * (0.5 - random.random())) * percent_BBOX_randomization
        max_y = max_y + (max_y - min_y) * (2 * (0.5 - random.random())) * percent_BBOX_randomization
        # RESTRICT MODIFIED BBOX BOUNDS FROM 0 TO 1
        min_x = max(0, min(1, min_x))
        max_x = max(0, min(1, max_x))
        min_y = max(0, min(1, min_y))
        max_y = max(0, min(1, max_y))

        image = image_processing.prepare_image(unmodified_camera_image, min_x, max_x, min_y, max_y)
        pose_data = item[0]
        local_angle = constants.relative_angle((min_x + max_x) / 2, pose_data[3])
        label_before_tensor[0] = pose_data[0]
        label_before_tensor[1] = pose_data[2]
        label_before_tensor[3] = math.cos(local_angle)
        label_before_tensor[2] = -math.sin(local_angle)
        label = torch.tensor(label_before_tensor).to(self.device)
        debug_var = 0
        if debug_var == 1:
            display_image_1 = image[0:3]
            plt.imshow(display_image_1.to("cpu").permute(1, 2, 0))
            plt.show()
            display_image_2 = image[3:4]
            plt.imshow(display_image_2.to("cpu").squeeze(0))
            display_image_3 = image[4:7]
            plt.imshow(display_image_3.to("cpu").permute(1, 2, 0))
            breakpoint_var = 1
        # self.image_cache[idx] = (image, label)
        return image, label


if __name__ == "__main__":
    dataset = CustomImageDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        test = 1
