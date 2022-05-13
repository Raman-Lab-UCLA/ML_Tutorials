import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cv2

# predict the spectrum from the model
from keras.models import load_model
model = load_model('/Users/kara-test/Desktop/UCLA/Raman Lab/Physics-drivenNN/model_interpolate_physics.h5', compile = False)

# load img
def loadImages(path):
    loadedImages = []
    # return array of images
    filesname = glob.glob(path)
    filesname.sort()
    for imgdata in filesname:
        # determine whether it is an image.
        if os.path.isfile(os.path.splitext(os.path.join(path, imgdata))[0] + ".png"): 
            img_array = cv2.imread(os.path.join(path, imgdata)) 
            img_array = np.float32(img_array)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) 
            loadedImages.append(gray)
    return loadedImages

# path = local path of images
path = '/Users/kara-test/Desktop/UCLA/Raman Lab/Physics-drivenNN/Training Data/Images_interpolate_train/*.png'
imgs = loadImages(path) # loaded images
imgs = np.array(imgs)
#imgs = np.squeeze(CNN_input_test)

for i in range(len(imgs)): 
    
    y = np.array(pd.read_csv('/Users/kara-test/Desktop/UCLA/Raman Lab/Physics-drivenNN/cross_length_interpolate_train.csv', header = 0, index_col=0))
    #y = CNN_output_test
    x = np.arange(4, 12, 0.01).reshape([1,800])
    predictions_test1 = model.predict(imgs[i].reshape(-1,128,128,1))
    print(predictions_test1)
    
    fig, ax = plt.subplots()
    ax.set(xlabel = 'Wavelength (nm)', ylabel = 'Absorption')
    ax.plot(x[0], y[i].reshape(1,800)[0][:], label = 'Ground Truth', linewidth = 4.0)    
    ax.plot(x[0], predictions_test1.reshape(1,800)[0][:], label = 'Predictions', linewidth = 4.0)  
    
    plt.rcParams['font.size'] = '18'
    plt.xlim(4,12)
    plt.ylim(0,1)
    #plt.legend()
    plt.savefig('/Users/kara-test/Desktop/UCLA/Raman Lab/Physics-drivenNN/test_results/'+str(i)+'.png', bbox_inches="tight")
    plt.show()