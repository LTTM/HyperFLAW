# HyperFLAW weather classifier

## Overview

This module allows you to train and test the weather classifier used in the [HyperFLAW](../) project. The classifier is designed to categorize weather conditions based on input data.

## Installation

This module is really simple so it is not necessary to install an ad-hoc environment just for this project. It is possible to run it directly from the [conda environment](../extras/environment.yml) used for the HyperFLAW project.

## Training the Classifier

To train the weather classifier, use the following command:

```bash
python train.py --num-epochs <num_epochs> --batch-size <batch_size> --lr <learning_rate> --model conv --dataset-path </path/to/dataset>
```

## Testing the Classifier

To test the weather classifier, use the following command:

```bash
python test.py --checkpoint-path </path/to/checkpoint> --dts-txt </path/to/dts-txt-splits>
```

## Example

Here is an example of how to train and test the classifier:

- Training:

    ```bash
    python train.py --num-epochs 8 --batch-size 88 --lr 0.0001 --model conv --dataset-path /data/dts/SynSELMA
    ```

- Testing:

    ```bash
    python test.py --checkpoint-path ./checkpoints/ConvolutionalConditionalClassifier_15-11-2023_00-08-56/model.pth --dts-txt ./data/test/acdc.txt
    ```

## Author

[Matteo Caligiuri](https://github.com/matteocali)
