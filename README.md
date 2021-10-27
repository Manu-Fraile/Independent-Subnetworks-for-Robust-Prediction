# Out Of Distribution detection using MIMO

## Install dependenices for original MIMO implementation
1. Activate a virtual environment by running `python3 -m venv venv`
1. Activate venv: `source venv/bin/activate`
    1. To deactivate venv: `deactivate`
1. Install dependencies: `pip install -r requirements.txt`
1. Install uncertainty_baselines dependency by running:
    1. pip install "git+https://github.com/google/uncertainty-baselines.git#egg=uncertainty_baselines"
    1. In order to get the cifar10 dataset locally, do:
        1. in `uncertainty_baselines/datasets/cifar.py` on row 109 add:
        1. `dataset_builder.download_and_prepare()`
    1. Alternatively, run `tfds build cifar10` in the terminal (??? check if this works!)
    1. TODO: get this stuff to work!! Have to download/store the datasets locally in some way..
1. Install robustness_metrics by running:
    1. pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics"
1. install Edward2 dependency by running `pip install edward2`

## Running the application
- with the virtual environment activated, run `python cifar.py`
- TODO: make this work!

## Library information:
- About the tensorflow_datasets module:
    - https://www.tensorflow.org/datasets/overview
