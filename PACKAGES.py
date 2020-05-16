import os
import cv2
import pickle
import numpy as np
import skimage
from skimage.feature import hog
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.svm import LinearSVC, SVC
import sklearn
import matplotlib.pyplot as plt
import random
import time
from pprint import pprint

