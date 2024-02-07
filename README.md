# Inhouse Dataset

This repository contains a dataset of images collected at Polo II, in the **Departamento de Engenharia Eletrotécnica e de Computadores** (DEEC) and  **Departamento de Engenharia e Informática** (DEI) at the **University of Coimbra** by the ISR Team. The dataset includes images in various modalities: *depth*, *intensity*, *RGB*, and *thermal*. Additionally, this repository also includes Python scripts for conversion and data augmentation.

To execute scripts, please install Albumentations and OpenCV
> pip install -U albumentations

> pip install opencv-python

## Script: YOLO-DATA-AUG.py

Description: This script reads a dataset in YOLO format, applies rotations and horizontal flips to the images for data augmentation, and then saves the augmented dataset in YOLO format.

Usage:
> python YOLO-DATA-AUG.py

Hint: Modify script to your custom dataset on line 147.

    if  __name__  ==  '__main__':


## Script: YOLO2COCO.py

Description: This script reads a dataset in YOLO format and converts it to the COCO format, saving the converted dataset.

Usage:

> python YOLO2COCO.py

Hint: Modify script to your custom dataset on line 182.

    if  __name__  ==  '__main__':
