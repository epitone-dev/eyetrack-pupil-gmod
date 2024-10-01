import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import cv2
import shutil

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class FacePupilDataset(Dataset):

    def __init__(self, face_pupil_dir, transform):
        self.face_pupil_dir = face_pupil_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # iterating through images directory 
        images_file = [file for file in os.listdir(face_pupil_dir)]

        # for each image in dataset folder
        for face_image in images_file:
            # join paths of image and dir
            if face_image.endswith('.jpg'):
                if face_image not in self.data:
                    image_path = os.path.join(face_pupil_dir, face_image)
                    self.data.append(image_path)
            if face_image.endswith('.txt'):
                if face_image not in self.labels:

                    # Join paths of label and directory
                    label_path = os.path.join(face_pupil_dir, face_image)

                    # Load label values from the .txt file
                    with open(label_path, 'r') as f:
                        # Read label data from the file
                        label_data = f.read().strip()

                        # Split the label values (assuming space-separated values in .txt files)
                        label_values = [float(value) for value in label_data.split()]

                        # Add label to the labels list
                        self.labels.append(label_values)

    def __len__(self):
        return len(self.labels)
    
    # need to crop images from original image to bounding box cropped image
    # need to modify corresponding labels for image so that new bounding box size is normalized 
    def __getitem__(self, index):
        
        image_path = self.data[index]
        label = self.labels[index]
        
        # loaded in the image and saved as PIL image format.
        image = cv2.imread(image_path)
        # crop image according to bounding box
        #label = torch.tensor(label, dtype=torch.float)

        img_height, img_width, _ = image.shape

        # bounding box labels extract from label
        bbox_cx = label[1]
        bbox_cy = label[2]
        bbox_w = label[3]
        bbox_h = label[4]

        right_cx = label[5]
        right_cy = label[6]
        left_cx = label[8]
        left_cy = label[9]


        bbox_left_cornerx = int((bbox_cx - (bbox_w/2))*img_width)
        bbox_right_cornerx = int((bbox_cx + (bbox_w/2))*img_width)
        bbox_top_cornery = int((bbox_cy - (bbox_h/2))*img_height)
        bbox_bottom_cornery = int((bbox_cy + (bbox_h/2))*img_height)

        image = image[bbox_top_cornery:bbox_bottom_cornery, bbox_left_cornerx:bbox_right_cornerx]
        #print(bbox_left_cornerx, bbox_right_cornerx, bbox_top_cornery, bbox_bottom_cornery)

        if self.transform:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            before_height, before_width, _ = image.shape
            image = Image.fromarray(image)
            image = self.transform(image)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            image = image * std + mean
            image = image.permute(1,2,0).numpy()
            image = (image*255).astype('uint8')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
        # Left pupil coordinates
        left_pupil_x = (((left_cx*img_width)-bbox_left_cornerx)/before_width)*400
        left_pupil_y = (((left_cy*img_height)-bbox_top_cornery)/before_height)*400

        # Right pupil coordinates 
        right_pupil_x = (((right_cx*img_width)-bbox_left_cornerx)/before_width)*400
        right_pupil_y = (((right_cy*img_height)-bbox_top_cornery)/before_height)*400
        
        coords = left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y


        set_data = {
            'image': image,
            'label': coords
        }
        print(set_data["label"])
        return set_data
    
    def __visualize__(self, index):
        
        image_path = self.data[index]
        label = self.labels[index]
        image = cv2.imread(image_path)

        img_height, img_width, _ = image.shape

        # bounding box labels
        bbox_cx = label[1]
        bbox_cy = label[2]
        bbox_w = label[3]
        bbox_h = label[4]
        # pupil position labels
        right_cx = label[5]
        right_cy = label[6]
        left_cx = label[8]
        left_cy = label[9]
        print(right_cx, right_cy, left_cx, left_cy)

        # bounding box corner calculations
        bbox_left_cornerx = int((bbox_cx - (bbox_w/2))*img_width)
        bbox_right_cornerx = int((bbox_cx + (bbox_w/2))*img_width)
        bbox_top_cornery = int((bbox_cy - (bbox_h/2))*img_height)
        bbox_bottom_cornery = int((bbox_cy + (bbox_h/2))*img_height)
        # pupil posiiton calculations


        # cropping to intervals
        cropped_img = image[bbox_top_cornery:bbox_bottom_cornery, bbox_left_cornerx:bbox_right_cornerx]
        image = cropped_img

        # bbox_height, bbox_width, _ = cropped_img.shape
        # print("-------------------BBOX SIZE-------------------")
        # print(bbox_height, bbox_width)
        
        # # Left pupil coordinates
        # left_pupil_x = left_cx*400-(bbox_left_cornerx)
        # left_pupil_y = left_cy*400-(bbox_top_cornery)

        # # Right pupil coordinates 
        # right_pupil_x = right_cx*400-(bbox_left_cornerx)
        # right_pupil_y = right_cy*400-(bbox_top_cornery)
        # print("---------LEFT PUPIL XY")
        # print(left_pupil_x, left_pupil_y)
        # print("---------RIGHT PUPIL XY")
        # print(right_pupil_x, right_pupil_y)

        # # Mark the left pupil on the cropped image (as a red dot)
        # cv2.circle(image, (int(left_pupil_x), int(left_pupil_y)), 2, (0, 0, 255), -1)  # Red dot with radius 5

        # # Mark the right pupil on the cropped image (as a blue dot)
        # cv2.circle(image, (int(right_pupil_x), int(right_pupil_y)), 2, (255, 0, 0), -1)  # Blue dot with radius 5


        # transforms activation if transform input
        if self.transform:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            before_height, before_width, _ = image.shape
            image = Image.fromarray(image)
            image = self.transform(image)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            image = image * std + mean
            image = image.permute(1,2,0).numpy()
            image = (image*255).astype('uint8')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bbox_height, bbox_width, _ = image.shape
        print("-------------------BBOX SIZE-------------------")
        print(bbox_width, bbox_height)
        
        print("-------------------before size-----------------")
        print(before_width, before_height)
        # Left pupil coordinates
        left_pupil_x = (((left_cx*img_width)-bbox_left_cornerx)/before_width)*400
        left_pupil_y = (((left_cy*img_height)-bbox_top_cornery)/before_height)*400

        # Right pupil coordinates 
        right_pupil_x = (((right_cx*img_width)-bbox_left_cornerx)/before_width)*400
        right_pupil_y = (((right_cy*img_height)-bbox_top_cornery)/before_height)*400

        print("---------LEFT PUPIL PRE")
        print(left_cx, left_cy, right_cx, right_cy)
        print("---------LEFT PUPIL XY")
        print(left_pupil_x, left_pupil_y)
        print("---------RIGHT PUPIL XY")
        print(right_pupil_x, right_pupil_y)


        img_height, img_width, _ = image.shape
        print(img_height, img_width)
        # Mark the left pupil on the cropped image (as a red dot)
        cv2.circle(image, (int(left_pupil_x), int(left_pupil_y)), 3, (0, 0, 255), -1)  # Red dot with radius 5

        # Mark the right pupil on the cropped image (as a blue dot)
        cv2.circle(image, (int(right_pupil_x), int(right_pupil_y)), 3, (255, 0, 0), -1)  # Blue dot with radius 5

        cv2.imshow('image', image)
        cv2.waitKey(0)
        return 0
    
# transforms defined
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.Resize((400,400))])

# running in directories and initializing dataloader
pupil_dir_tr = "../../Desktop/sample_data_pupil"
test_set = FacePupilDataset(pupil_dir_tr, transform=transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# testing code here
a = test_set.__getitem__(0)
print(a)
test_set.__visualize__(0)