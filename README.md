# Unveiling Urban Parking: Enhanced Image Analysis for Vacant Space Detection

Forked from official repository for the [Image-Based Parking Space Occupancy Classification: Dataset and Baseline](https://arxiv.org/abs/2107.12207) paper.

We introduce an effective training method using a custom data augmentation strategy to improve previous models' abilities to reason about occlusions caused surrounding vehicles, trees, and other natural or urban features. 

In this repository, we provide:
- Code to reproduce all results.
- Download link for the [dataset](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip).
- Notebooks to [explore the dataset](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/acpds_playground.ipynb), [trial and visualize a variety of augmentations](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/augmentation_playground.ipynb), [train model](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/train.ipynb), [and visualize/ reproduce results](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/results.ipynb).
- [Scripts](https://github.com/ethanclewis/parking-space-occupancy/tree/main/utils) to alter elements of the model training process.

# Dataset

The dataset (*Action-Camera Parking Dataset*) contains 293 images captured at a roughly 10-meter height using a GoPro Hero 6 camera. Here is a sample from the dataset:

![alt text](illustrations/dataset_sample.jpg)

# Results

To reproduce our quantitative results, simply clone the repo and run the [results notebook](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/results.ipynb) locally. 

# Training

To understand the workflow of the repo, train your own models, etc., we recommened working through the notebooks in the following order:
- [acpds_playground](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/acpds_playground.ipynb)
- [roi_analysis](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/roi_analysis.ipynb)
- [augmentation_playground](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/augmentation_playground.ipynb)
- [train](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/train.ipynb)
- [results](https://github.com/ethanclewis/parking-space-occupancy/blob/main/notebooks/results.ipynb)

# Citation

```bibtex
@misc{marek2021imagebased,
      title={Image-Based Parking Space Occupancy Classification: Dataset and Baseline}, 
      author={Martin Marek},
      year={2021},
      eprint={2107.12207},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
