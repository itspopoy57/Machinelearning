
import zipfile
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip

#unzip
zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip" , "r")
zip_ref.extractall()
zip_ref.close()

from posixpath import dirname
import os

for dirpath, dirnames, filenames in os.walk("10_food_classes_all_data"):
  print(f"there are {len(dirnames)} directories and {len(filenames)} files in the directory {dirpath}")

train_dir_full = "/content/10_food_classes_all_data/test/"
test_dir_full = "/content/10_food_classes_all_data/train/"

#lets get the class names
import pathlib
import numpy as np

data_dir = pathlib.Path(train_dir_full)
class_name = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_name)


# View an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img


# View a random image from the training dataset
img = view_random_image(target_dir = train_dir_full,
                        target_class=random.choice(class_name))
