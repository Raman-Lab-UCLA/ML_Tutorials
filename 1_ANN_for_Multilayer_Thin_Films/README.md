A simple fully-connected neural network or multilayer perceptron (MLP) for the analysis of multi-layered thin film stacks. Inputs are layer thicknesses (specified in the PPT) and outputs are the points in a reflection spectrum.

## Requirements
For convenience, here are installation commands for the Conda distribution (after installing Anaconda: https://www.anaconda.com/products/individual).

```
conda create -n myenv python=3.7
conda activate myenv
conda install tensorflow
conda install -c anaconda scikit-learn
conda install matplotlib
conda install pandas
conda install spyder
```
## Notes
First time users of Spyder may run into issues of storing/referencing variables between scripts/runs. To fix this, go to Tools>Preferences>Run and tick in the box for "Run in consoles namespace instead of an empty one".
