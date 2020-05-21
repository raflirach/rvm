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