# Over-the-Air Ensemble Inference with Model Privacy

This repository contains simulations for our private ensemble inference method.

### Installation
* Install conda and torch manually (recommended)
* `pip install -r requirements.txt`

### Running
* First train and cache the device models.
* Then you can generate figures, tables or run raw experiments.

### Training CV models
* `python train.py --data <data_name> --num_repeats 10 --num_devices 20 --num_epochs 50`
* `<data_name>` can be `cifar10`, `cifar100`, `mnist`, `fashionmnist`

### Training NLP models
* `python nlp_train.py --data <data_name> --num_repeats 10 --num_devices 20`
* `<data_name>` can be `yelp_review_full`, `yelp_polarity`, `imdb`, `emotion`

### Running an Experiment
* See the bottom of `ota_private_ensemble.py`

### Generating TeX Code for the Comparison table
* Run `python figure_comparison_table.py`

### Generate TeX Code for the Varying Conditions pgfplot
* Run `python figure_conditions.py`
