import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cv2
import autokeras as ak

# predict the spectrum from the model
from tensorflow.keras.models import load_model
model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

# load img
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

# path = local path of images
path = 'C:/Users/ramanlab/Desktop/Kara/Explainability_for_Photonics-master/Training Data/Images/*.png'
imgs = loadImages(path)
imgs = np.array(imgs)

for i in range(len(imgs)): 
    y = np.array(pd.read_csv('C:/Users/ramanlab/Desktop/Kara/Explainability_for_Photonics-master/Training Data/Spectra.csv', header = 0, index_col=0))
    x = np.arange(4, 12, 0.1).reshape([1,80])
    predictions_test1 = model.predict(imgs[i].reshape(-1,imgs.shape[1],imgs.shape[1],1))
    print(predictions_test1)
    fig, ax = plt.subplots()
    ax.set(xlabel = 'Wavelength (µm)', ylabel = 'Absorption')
    ax.plot(x[0], y[i].reshape(1,80)[0][:], label = 'Ground Truth', linewidth = 4.0)    
    ax.plot(x[0], predictions_test1.reshape(1,80)[0][:], label = 'Predictions', linewidth = 4.0)  
    plt.rcParams['font.size'] = '18'
    plt.xlim(4,12)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('C:/Users/ramanlab/Desktop/Kara/Explainability_for_Photonics-master/Test_Results/'+str(i)+'.png', bbox_inches="tight")
    plt.show()