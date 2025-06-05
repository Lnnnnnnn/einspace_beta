#!/usr/bin/env python3
# Downloads & checks all datasets

print("Loading code...")
from data import get_data_loaders

print("Checking datasets...")
datasets = ["cifar10", "cifar100", "addnist", "language", "multnist", "cifartile", "gutenberg", "isabella", "geoclassing", "chesseract"]
batch_size = 64
image_size = 28
for dataset in datasets:
    print(f"Checking dataset '{dataset}'...")
    print(get_data_loaders(dataset, batch_size, image_size, download=True))
