# Consistent time reversal and reliable and accurate inference in the presence of memory

This code contains GPU-optimised code for a higher order estimator used in Fig. 1. 
Go through `six-state-example.ipynb` for a hands-on example on how to use the framework and for comments on certain key parameters.

Note that this version repository contains the *GPU optimised* code, so we recommend running this on a single GPU like H100.

## Installation

Install Miniconda, check that your  cuda version matches the one in environment.yml (currently 12, you may update that in environment.yml if your system is newer). Then run from the repository root:

```bash
conda env create -f environment.yml
conda activate consistent-time-reversal-estimator-framework
```


Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user \
    --name consistent-time-reversal-estimator-framework \
    --display-name "Python (consistent time reversal estimator framework)"
```

Then start the notebook  `six-state-example.ipynb` and select the kernel named Python (consistent time reversal estimator framework).