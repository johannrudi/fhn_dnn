# Inference using Deep Neural Networks applied to FitzHugh-Nagumo ODE

The goal is to estimate parameters in ordinary differential equations (ODEs) from data that represents the output of an ODE.  We perform such an inference using a trained Deep Neural Network (DNN).  The particular ODE targeted here is the FitzHugh-Nagumo, which models voltage spikes of biological neurons.

This code demonstrates some of the numerical experiments of the paper:

**Parameter Estimation with Dense and Convolutional Neural Networks Applied to the FitzHugh-Nagumo ODE** 
by Johann Rudi, Julie Bessac, Amanda Lenzi, 2021. 
URL: https://arxiv.org/abs/2012.06691

It was published in the proceedings of the conference for Mathematical and Scientific Machine Learning (MSML21).

## Running the code

The executable code is located in the directory `tensorflow`.
To run, execute the script `run_dnn.py`.
Different options for the inference can be set by modifying the file `config/params.yaml`.

The data for the training and evaluating the trained DNN models is located in the directory `data/float32`.

## Imported Python packages

It is required to install the packages:

- matplotlib
- numpy
- sklearn
- tensorflow

Additionally, the following standard packages are used:

- argparse
- os
- pathlib
- pprint
- time
- sys
- yaml
