# Out Of Distribution detection using MIMO
Deep neural networks can help to solve complex tasks, but understanding the processes within neural networks and estimating the uncertainty of predictions is
difficult. The MIMO architecture introduces a method that adds minimal additional computational cost compared to other techniques for uncertainty estimation. In
this project we reproduce some of the key results from the original MIMO paper, in some cases using untested architecture changes and datasets. Additionally, we
conduct new experiments on OOD detection where we use predictive entropy to evaluate the quality of the uncertainty estimates. Our results indicate that the results
obtained in the original paper are reproducible, and also give promising results on our new experiments.

This work received the maximum possible grade, an A.


## Install dependenices for original MIMO implementation
1. Activate a virtual environment by running `python3 -m venv venv`
1. Activate venv: `source venv/bin/activate`
    1. To deactivate venv: `deactivate`
1. Install dependencies: `pip install -r requirements.txt`
1. Install uncertainty_baselines dependency by running:
    1. pip install "git+https://github.com/google/uncertainty-baselines.git#egg=uncertainty_baselines"
1. Install robustness_metrics by running:
    1. pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics"
1. install Edward2 dependency by running `pip install edward2`

## Running the application
THIS IS NOT WORKING ALL THE WAY - still error on row 225 in cifar.py

1. Go to `/mimo_code/` directory
1. with the virtual environment activated, run `python cifar.py`
1. The first time the code is run, it will download associated datasets which can take a while. It is possible that problems are encountered in this process, here are a few things to try if there are errors:
    1. Remove the virtual environment and re-install everything
    1. Add code in the uncertainty_baselines package:
        1. in `uncertainty_baselines/datasets/cifar.py` on row 109 add:
        1. `dataset_builder.download_and_prepare()`
    1. run `tfds build cifar10` in the terminal

## Library information:
- About the tensorflow_datasets module:
    - https://www.tensorflow.org/datasets/overview
