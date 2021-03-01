# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:22:09 2020

@author: jonherr21
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as tb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import requests
import sys
import math

from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask,request
from datetime import datetime

class Server(object):
    
    app = None

    def __init__(self):
    
        self.app = Flask(__name__)
        tb._SYMBOLIC_SCOPE.value = True

        self.model_RF = pickle.load(open("model_RF.pkl","rb"))
        self.model_NN = load_model(open("model_NN.h5","rb"))
        self.scaler = pickle.load(open("scaler.pkl","rb"))  
        
        
        @self.app.route('/',methods=['POST'])
        def API():
            data = request.form.to_dict(flat=False)
            
            diag = []
            diag.append(float(data.get("hematocrit")[0]))
            diag.append(float(data.get("hemoglobin")[0]))
            diag.append(float(data.get("platelets")[0]))
            diag.append(float(data.get("mean_platelet_volume")[0]))
            diag.append(float(data.get("red_blood_cells")[0]))
            diag.append(float(data.get("lymphocytes")[0]))
            diag.append(float(data.get("mean_corpuscular_hemoglobin_concentration_mchc")[0]))
            diag.append(float(data.get("leukocytes")[0]))
            diag.append(float(data.get("basophils")[0]))
            diag.append(float(data.get("mean_corpuscular_hemoglobin_mch")[0]))
            diag.append(float(data.get("eosinophils")[0]))
            diag.append(float(data.get("mean_corpuscular_volume_mcv")[0]))
            diag.append(float(data.get("red_blood_cell_distribution_width_rdw")[0]))

            result = ''
            
            if data.get("diagnosis")[0]=='random forest':
                result = self.model_RF.predict(np.array(diag).reshape(1,-1))
                result = (str(result[0]))
                result = 'covid '+result
                
            if data.get("diagnosis")[0]=='neural network':
                x_nn = self.scaler.transform(np.array(diag).reshape(1,-1))
                result = self.model_NN.predict_classes(x_nn)
                if result==0:
                    result = 'covid negative'
                else:
                    result = 'covid positive'

            return result