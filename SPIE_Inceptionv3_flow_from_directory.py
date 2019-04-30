from predprob import predprob
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
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
import matplotlib.pyplot as plt
import scipy.stats as stats

if(0):
    pathPrefix = "F:\\Studie\\"
else:
    pathPrefix = "C:\\Users\\s155868\\"
import os
os.chdir(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP")




############
### Import the main dataset
############


#Annotations
anno = pd.read_csv(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\train_labels.csv")
trainCellularity=[]
for i in range(len(images)):
    trainCellularity.append(float(anno.loc[operator.and_(anno['slide']==int(image_slide[i]), anno['rid']==int(image_region[i]))]['y']))

anno = pd.read_csv(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\val_labels.csv")
valCellularity=[]
for i in range(len(images)):
    valCellularity.append(float(anno.loc[operator.and_(anno['slide']==int(image_slide[i]), anno['rid']==int(image_region[i]))]['y'])) 
        
#Make everything in array type
trainCellularity = np.array(trainCellularity)
valCellularity = np.array(valCellularity)

#Check the division of the cellularity score over train/ind/test
print(np.mean(trainCellularity)) #train mean:0.30, median:0.20
print(np.mean(valCellularity)) #validation mean:0.37, median:0.30


print("Loaded the dataset.")

############
### Pre-trained model
###########

# Create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- sigmoid to make sure all predictions lie between 0 and 1
predictions = Dense(1, activation='sigmoid')(x)

# First: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# I chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
#for layer in model.layers[:41]:
#   layer.trainable = False

#Here I chose to train all layers
#for layer in model.layers[:]:
#   layer.trainable = True

# Compile the model (should be done *after* setting layers to non-trainable)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['mean_squared_error'])

# Build a data augmentor
datagen =  ImageDataGenerator(
        rotation_range=np.pi,
        width_shift_range=0.25,
        height_shift_range=0.25,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        channel_shift_range=15) 
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\train", 
                                        np.reshape(trainCellularity, (-1,1)), #DIT KAN NIET! Je kan niet je eigen labels toevoegen bij flow_from_directory
                                        batch_size=10, 
                                        shuffle=True)

val_gen = val_datagen.flow_from_directory(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\validation", 
                                          np.reshape(cellularity[valind], (-1,1)), 
                                          batch_size=10, 
                                          shuffle=False)

# checkpoint
filepath=pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\models\\inceptionv3_layer41_patient.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
callbacks_list = [checkpoint, tensorboard]
   
datagen.flow(images[trainind], np.reshape(cellularity[trainind], (-1,1)), batch_size=10, shuffle=True)

print("Starting initial training.")
# train the model on the new data for a few epochs
model.fit_generator(, steps_per_epoch=100, epochs=100, validation_data=, validation_steps=22, callbacks=callbacks_list)

#Load the best weights in the model
model.load_weights(filepath)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

# I chose to train a lot of the inception blocks, i.e. we will freeze
# the first 41 layers and unfreeze the rest:
for layer in model.layers[:41]:
   layer.trainable = False
for layer in model.layers[41:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use adam with a low learning rate
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['mean_squared_error'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(datagen.flow(images[trainind], np.reshape(cellularity[trainind], (-1,1)), batch_size=10, shuffle=True), steps_per_epoch=100, epochs=100, validation_data=val_datagen.flow(images[valind], np.reshape(cellularity[valind], (-1,1)), batch_size=10, shuffle=False), validation_steps=25, callbacks=callbacks_list)

#Load the best weights in the model
model.load_weights(filepath)

###########
### Predict with trained model
###########

# Apply model and see how well it does on the validation set
pred = model.predict(images/255)
def round_nearest(x, a):
    return np.round(x / a) * a
round_pred = round_nearest(pred,0.05)
#Check the mse or tau-b score
###Write code here to evaluate the classifier
#The predprob function comes directly from the challenge organizers
pred_prob = predprob(cellularity, pred)
tau_b, p_value = stats.kendalltau(pred, cellularity[valind])
np.savetxt("SPIE_truth_val.csv", cellularity, fmt='%1.18f', delimiter=',')

#Plot
plt.scatter(cellularity[valind], pred)
plt.xlabel("Ground truth")
plt.ylabel("Model prediction")

#Make nice results table
plain = pd.DataFrame()
plain['slide'] = image_slide
plain['image'] = image_region
plain['prediction_plain'] = pred
plain['truth'] = cellularity

#Add nuclei results to my table
nuclei = pd.read_csv(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\nuclei_results.csv", sep='\s*,\s*')
nuclei['slide'] = nuclei['slide'].astype(np.int64)
nuclei['image'] = nuclei['image'].astype(str)
merge =pd.merge(plain, nuclei[['slide','image','prediction']], on=['slide','image'])

pred_avg = np.mean((np.array(merge['prediction_plain']),np.array(merge['prediction'])), axis=0)
round_pred_avg = round_nearest(pred_avg,0.05)
tau_b_avg, p_value = stats.kendalltau(round_pred_avg[valind], cellularity[valind])
