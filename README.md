# cvn
Modeling exercise for the Computational Visual Neuroscience seminar HS21:

`Generate yourself some data from a simple 
(polynomial) model with noise. Then write a program in R, Python or 
MATLAB to do the three essential modeling steps: optimisation, goodness-of-
fit and estimating variability (using the correct, generative model).`

This application thus consists of a jupyter notebook, and a few python objects which together implement a framework to generate a sample dataset, and a model to be fitted to it using a polynomial function of arbitrary order.

### how to run
To run the notebook, `jupyter notebook` needs to be installed, for example with `pip`:

    pip install notebook

Then, in the `cvn` directory, run 

    jupyter notebook cvn.ipynb

A browser window should open, in which the notebook can be run. Dependencies should be installed automatically in the first code block, if not, use `pip` to install:

- numpy
- scipy
- matplotlib

then re-run the notebook.