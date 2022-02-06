# TLI Preprocessing Pipeline

This is a python pipeline for preprocessing tif images.
The complete pipeline can do the following preprocessing steps:
1. compile: compiling 3D into a single 4D images (array). The 4D images created based on the specified channels and saved into a dictionary
2. preshift: applying phase correlation on a dictionary of 4D images based on the first dimension (time) and the specified reference channel
3. trim: trimming the last quarter of the images in the second dimension (Z) to reduce runtime in subsequent steps
4. ants: applying multi-step Antspy registration on a dictionary of 4D images based on the last channel of the 4D_image dictionary
4. n2v: applying n2v denoising on the first 4D image element (channel) of a 4D images dictionary 
5. clahe: applying clahe contrast-enhancement on the first 4D image element (channel) of a 4D images dictionary 
6. mask: creating a mask of the first 4D image element (channel) of a 4D images dictionary 
7. segment: segmenting the first 4D image element (channel) of a 4D images dictionary into dictionary of single 4D image-segments (neurons)
8. postshift: applying multi-step Antspy registration on a dictionary of 4D images based on the first channel of the 4D_image dictionary

## Installation

The script was created based on python3.8, and requires [Tensorflow](https://www.tensorflow.org/install/).
First the puplically-available pipline can be cloned from github using 
```bash
pip3 install git+https://https://github.com/Asashou/TLI/
```
An environment with the required packages has to be created using the provided environment.yml file
```bash
conda env create -f environment.yml
```
## Usage

The script reads an info text file that provides all the required input variables like the path to data and save_path.<br/>
an example of the pipelin_info.text can be find in the repository <br/>
The pipeline can be started then in the bash terminal<br/>

```bash
python /scripts/general_pipline_4D.py /scripts/general_pipline_info.txt
