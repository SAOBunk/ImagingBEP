from predprob import predprob
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
import numpy as np
import glob
from PIL import Image
import re
import pandas as pd
import operator
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import scipy.stats as stats

############
### Pre-trained model
###########

# Create the base pre-trained model
base_model = ResNet50(weights=None, include_top=False)#, input_shape=(512,512,3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- sigmoid to make sure all predictions lie between 0 and 1
predictions = Dense(1, activation='sigmoid')(x)


#Change the momentum for the good of the people
for i, layer in enumerate(base_model.layers):
	name = str(layer.name)
	#print(name[0:2])
	if(name[0:2]=="bn"):
		config = layer.get_config()
		config['momentum'] = 0.01 
		base_model.layers[i] = BatchNormalization.from_config(config)
		
# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# Compile the model (should be done *after* setting layers to non-trainable)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['mean_squared_error'])

print("Compiled model.")

#%% let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
	name = str(layer.name)
	if(name[0:2]=="bn"):
		config = layer.get_config()
		print(i, config)
