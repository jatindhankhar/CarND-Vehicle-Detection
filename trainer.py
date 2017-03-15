import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle
import cv2
import numpy as np
#%matplotlib qt
from lesson_functions import *

cars = glob('vehicles/*/*.png',recursive=True)
non_cars = glob('vehicles/*/*.png',recursive=True)

cars_labels = np.ones(len(cars))
non_cars_labels = np.zeros(len(non_cars))
random_idx = np.random.randint(0,len(cars))

print("Extracting Features ...")
cars_features = extract_features(cars,hog_channel='ALL')
non_cars_features = extract_features(non_cars,hog_channel='ALL')
print("Done.. ")
