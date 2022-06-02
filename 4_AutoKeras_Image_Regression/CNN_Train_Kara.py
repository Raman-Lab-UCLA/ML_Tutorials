import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob
import cv2
import autokeras as ak
import time
import sys

oldStdout = sys.stdout # = output file object, directly displays everything written to it to the console
file = open('logFile.txt', 'w')
sys.stdout = file

## Define File Locations (Images, Spectra, and CNN Model Save)
img_path = 'C:/Users/ramanlab/Desktop/Kara/Explainability_for_Photonics-master/Training Data/Images/*.png'
spectra_path = 'C:/Users/ramanlab/Desktop/Kara/Explainability_for_Photonics-master/Training Data/Spectra.csv'
save_dir = 'C:/Users/ramanlab/Desktop/Kara/Explainability_for_Photonics-master/saved models/saved models/model.h5'

## Load Images (CNN Input)
def loadImages(path):
    loadedImages = []
    filesname = glob.glob(path)
    filesname.sort()
    for imgdata in filesname:
        if os.path.isfile(os.path.splitext(os.path.join(path, imgdata))[0] + ".png"):
            img_array = cv2.imread(os.path.join(path, imgdata))
            img_array = np.float32(img_array)
            img_size = 40
            new_array = cv2.resize(img_array, (img_size, img_size))
            gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)       
            loadedImages.append(gray)
    return loadedImages

imgs = loadImages(img_path)
CNN_input = np.array(imgs).reshape(len(imgs),40,40,1)

## Load Spectra from Excel (CNN Output)
CNN_output = np.array(np.float32(pd.read_csv(spectra_path, header = 0, index_col=0)))

# Split Data into Train and Test Sets
CNN_input_train, CNN_input_test, CNN_output_train, CNN_output_test = train_test_split(CNN_input, CNN_output, test_size = 0.1, random_state = 42)
print('data size after spliting')
print('CNN_input_train size: {}'.format(np.shape(CNN_input_train)))
print('CNN_input_test size: {}'.format(np.shape(CNN_input_test)))
print('CNN_output_train size: {}'.format(np.shape(CNN_output_train)))
print('CNN_output_test size: {}'.format(np.shape(CNN_output_test)))

# Train for X Trials
##Custom Search Space (Restricts Search Space to Specific CNN Types) - Comment/Uncomment Here to Use
input_node = ak.ImageInput()
output_node = ak.ImageBlock(block_type="vanilla", normalize=False, augment=False)(input_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=10)

##Non-Custom Search Space - Comment/Uncomment Here to Use
# reg = ak.ImageRegressor(max_trials=10, overwrite=True)

reg.fit(CNN_input_train, CNN_output_train, epochs=300)   #see if you can find doc on what last "train" is
score = reg.evaluate(CNN_input_test, CNN_output_test)
print('val_loss:', score[0])


model = reg.export_model()
model.save("model_autokeras", save_format="tf")
model.summary()
time.sleep(10)
sys.stdout = oldStdout

##Retrieves Best Models
# reg.tuner.get_best_models(5)
# reg.tuner.results_summary()
