import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cv2

# predict the spectrum from the model
from keras.models import load_model
model = load_model('/.../model.h5', compile = False)

# load img
def loadImages(path):
    loadedImages = []
    filesname = glob.glob(path)
    filesname.sort()
    for imgdata in filesname:
        if os.path.isfile(os.path.splitext(os.path.join(path, imgdata))[0] + ".png"): 
            img_array = cv2.imread(os.path.join(path, imgdata)) 
            img_array = np.float32(img_array)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) 
            loadedImages.append(gray)
    return loadedImages

# path = local path of images
path = '/.../Images_test/*.png'
imgs = loadImages(path)
imgs = np.array(imgs)

for i in range(len(imgs)): 
    y = np.array(pd.read_csv('/.../cross_length_test.csv', header = 0, index_col=0))
    x = np.arange(4, 12, 0.01).reshape([1,800])
    predictions_test1 = model.predict(imgs[i].reshape(-1,imgs.shape[1],imgs.shape[1],1))
    print(predictions_test1)
    fig, ax = plt.subplots()
    ax.set(xlabel = 'Wavelength (µm)', ylabel = 'Absorption')
    ax.plot(x[0], y[i].reshape(1,800)[0][:], label = 'Ground Truth', linewidth = 4.0)    
    ax.plot(x[0], predictions_test1.reshape(1,800)[0][:], label = 'Predictions', linewidth = 4.0)  
    plt.rcParams['font.size'] = '18'
    plt.xlim(4,12)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('/.../Test_Results/'+str(i)+'.png', bbox_inches="tight")
    plt.show()
