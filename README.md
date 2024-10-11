# Fire-Detection

This project implements two approches of a deep-learning based fire detection system. It is is designed to detect fire in images, using a YOLOv5 or a smaller multi-head detection model. The project is implemented in PyTorch, and is designed to be easily extendable to other models and datasets.

## Project Architecture

* [YOLO Notebook](yolo.ipynb) - A Jupyter notebook that trains the YOLOv5 pretrained model for fire detection.
* [Multi-Head Notebook](multi_head.ipynb) - A Jupyter notebook that trains a smaller multi-head model for fire detection. It also shows the processing applied to the images before training.
* [Script files](src/) - Modularized Python scripts that implement, datasets, preprocessing, models and training. They are meant to be imported in the notebooks.
* [Report](report.pdf) - A report detailing the project, methodoly and some experimental results.
* [Requirements](requirements.txt) - A list of the required packages to run the project, that can be installed with `pip install -r requirements.txt`.