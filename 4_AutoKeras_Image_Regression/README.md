Code for running AutoML (AutoKeras) for CNN model optimization.

## Requirements
For convenience, here are installation commands for the Conda distribution (after installing Anaconda: https://www.anaconda.com/products/individual).

```
conda create -n autokeras python=3.7
conda activate autokeras
conda install -c anaconda opencv
pip install autokeras 
conda install -c anaconda scikit-learn
conda install matplotlib
conda install spyder
```
NOTE: AutoKeras is constantly updated and may depend on newer versions of Tensorflow, which depend on specific versions of CUDA and cuDNN. Refer to the following for compatibility details: https://www.tensorflow.org/install/source#gpu.
