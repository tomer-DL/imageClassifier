import numpy as np
import os
import cv2
from read_ini import read_section

def file_generator(root, ext=""):
    for x in os.listdir(root):
        if os.path.isfile(root + "/" + x) and x.endswith(ext):
            yield root + "/" + x


