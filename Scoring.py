#Scoring of neural networks with a spectrum of metrics
from predprob import predprob
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.stats as stats
import operator
import re

if(1):
    pathPrefix = "F:\\Studie\\"
else:
    pathPrefix = "C:\\Users\\s155868\\"
import os
os.chdir(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP")


############
### Import the main dataset
############

#Images
images=[]
image_slide=[]
image_region=[]
for img in sorted(glob.glob(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\train\\*.tif")):
    image_slide.append(img[(68+len(pathPrefix)):(73+len(pathPrefix))])
    image_region.append(re.sub("\D", "", img[-6:-4]))

#Annotations
anno = pd.read_csv(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\train_labels.csv")
cellularity=[]
for i in range(len(image_slide)):
    cellularity.append(float(anno.loc[operator.and_(anno['slide']==int(image_slide[i]), anno['rid']==int(image_region[i]))]['y']))
 
patient_id = pd.read_csv(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\patient_ids.csv")

#Create train/val/test set based on patient id
unique = list(set(patient_id['patient_id'])) #63 unique WSIs here --> 45 train / 8 val / 10 test
np.random.seed(seed=12)
train_id, validation_id, test_id = np.split(np.random.choice(unique, size=len(unique), replace=False), [int(.7*len(unique)), int(.85*len(unique))])  

#Patient_id to WSIs
train=[]
for i in range(len(train_id)):
    for j in range(len(patient_id['slide'][patient_id['patient_id']==train_id[i]])):
        loc = np.where(patient_id['patient_id']==train_id[i])[0][j]
        train.append(patient_id.iloc[loc,1])
validation=[]
for i in range(len(validation_id)):
    for j in range(len(patient_id['slide'][patient_id['patient_id']==validation_id[i]])):
        loc = np.where(patient_id['patient_id']==validation_id[i])[0][j]
        validation.append(patient_id.iloc[loc,1])
test=[]
for i in range(len(test_id)):
    for j in range(len(patient_id['slide'][patient_id['patient_id']==test_id[i]])):
        loc = np.where(patient_id['patient_id']==test_id[i])[0][j]
        test.append(patient_id.iloc[loc,1])

image_slide = np.array(image_slide).astype('int')

trainind=[]
valind=[]
testind=[]
for i in range(len(image_slide)):
    if (image_slide[i] in test):
        testind.append(i)
    elif (image_slide[i] in validation):
        valind.append(i)
    else:
        trainind.append(i)
		
		
#Make everything in array type
cellularity = np.array(cellularity)

#Check the division of the cellularity score over train/ind/test
print(np.mean(cellularity[trainind])) #train mean:0.30, median:0.20
print(np.mean(cellularity[valind])) #validation mean:0.37, median:0.30
print(np.mean(cellularity[testind])) #test mean:0.35, median:0.25

print("Loaded the dataset.")


#Predictions
pred_inceptionv3 = np.array(pd.read_csv(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\predictions\\Inceptionv3_predictions.csv", header=None))
pred_vgg19 = np.array(pd.read_csv(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\predictions\\VGG19_predictions.csv", header=None))
#pred_xception = np.array(pd.read_csv(pathPrefix+"OneDrive - TU Eindhoven\\Vakken\\2018-2019\\Kwart 4\\BEP\\datasets\\predictions\\Xception_predictions.csv", header=None))

predictions = [pred_inceptionv3, pred_vgg19]#, pred_xception]
names = ["Inceptionv3", "VGG19", "Xception"]

print("loaded prediction data.")

#%% Perform statistical analysis
for i,pred in enumerate(predictions):
	pred_prob = predprob(cellularity[testind], pred[testind])
	print("Prediction probability score for {0}: {1}".format(names[i],pred_prob))
	tau_b, p_value = stats.kendalltau(pred[testind], cellularity[testind])