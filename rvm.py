import matplotlib.pyplot as plt
import numpy as np
from sklearn_rvm import EMRVC
import warnings
from abc import ABCMeta, abstractmethod

import os
import math
import cv2
import numpy as np
import scipy.linalg
from numpy import linalg
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from flask import Flask, request, jsonify, render_template
from flask import request
from flask_cors import CORS

app = Flask(__name__, template_folder='template')

def load_images(folder):
    images = np.array([])
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
#         img = cv2.resize(img, (160, 200))
        im_canny = cv2.Canny(img,75,150)
#         retval, im_biner = cv2.threshold(im_canny, 127, 1, cv2.THRESH_BINARY)
        moments = cv2.moments(im_canny)
        huMoments = cv2.HuMoments(moments)
        for i in range(0,7):
            huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        if huMoments is not None:
            images = np.append([[images]], huMoments)
    images = images.reshape(int(len(images)/7),7)        
    return images

imgdata = load_images('dataset/data3')
print(imgdata)

@app.route('/')
def home():
    return render_template("index.html", imgdata=imgdata)

if __name__ ==  "__main__":
    app.run()