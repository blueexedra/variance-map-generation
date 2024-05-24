# VarianceMapGeneration
## Introduction
This repository contains programs for generating variance maps (images in which the pixel values represent the variance within a window centered on the region of interest).

## Prerequisites
- gcc (supports C++17)
- cmake (ver 3.9 or higher)
- OpenCV (ver 3.X or higher)
- OpenMP

## Installation
Use following commands.
```
git clone https://github.com/blueexedra/variance-map-generation.git
cd variance-map-generation/src
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage
Simply execute built binary, and you will get following message.
```
Usage: variance_map [image_path] [window_size]
```
"image_path" is the path to input images. The variance maps are output to "image_path/variance/".