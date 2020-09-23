*"Give me six hours to chop down a tree, and I will spend the first four sharpening the axe."* 

--Abraham Lincoln.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python-code)
    + [channel](#channel)
    + [decoders](#decoders)
    + [trainers](#trainers)
    + [plotting](#plotting)
    + [utils](#utils)
    + [config](#config)
  * [resources](#resources)
  * [dir_definitions](#dir-definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)
- [Citation](#citation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


# Introduction

This repository simulates the building blocks of a simplified communication diagram, focusing on the implementation of several classical and machine-learning based decoders for tail-biting convolutional codes. It explores several models: CVA, WCVA, WCVAE, gated WCVAE. The CVA is the circular Viterbi algorithm from [this paper](https://ieeexplore.ieee.org/document/282266). The WCVA is the weighted CVA. The WCVAE stands for the weighted circular Viterbi algorithm ensemble. At last, the gated WCVAE lowers the complexity of the WCVAE by employing a low-complexity gating. All methods are explained in the [corresponding paper](https://arxiv.org/abs/2009.02591).

# Folders Structure

## python_code 

The python simulations of the encoder & channel and decoders.

### channel 

Includes the channel model and BPSK modulator. The main function is the channel dataset - a wrapper of the ground truth codewords, along with the received channel word. Used for training the decoders as well as evaluation.

### decoders

The backbone decoders: CVA, WCVA, WCVAE, gated WCVAE.

### trainers 

Wrappers for training and evaluation of the decoders. 

Each trainer inherets from the basic trainer class, extending it as needed.

### plotting

Plotting of the FER versus SNR, and the FER versus the states. See Figures 4-6 in the paper.

### utils

Extra utils for saving and loading pkls; tail-biting related calculations; and more...

### config

Controls all parameters and hyperparameters in Tables I and II.

## resources

Folder with all codes matrices.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 2060 with driver version 432.00 and CUDA 10.1. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f deep_ensemble.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the deep_ensemble environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\deep_ensemble\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!

# Citation

Please cite our [paper](https://arxiv.org/abs/2009.02591) if the code is used for publishing research.

Credit to [this repo](https://github.com/SharonLK/psagot-2020-algo) for the environment installation.
