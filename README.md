# Smoker Detection (CNN / Transfer Learning)

An image classification exploration: detecting smoker vs. non-smoker from facial images using a transfer-learning CNN (InceptionResNetV2) on a Kaggle dataset.

## Approach

- Dataset split 80/20 into train/validation, organized into `Smoker` / `Non-Smoker` classes
- Faces cropped from source images before training (`roi_dataset`)
- InceptionResNetV2 (ImageNet weights) as a frozen feature extractor with a classification head trained on top
- Evaluated with scikit-learn metrics

## Tech Stack

TensorFlow/Keras, OpenCV, scikit-learn, Python.

## Status

The training notebook itself isn't currently in this repo — only the write-up and one output image survived. Worth re-adding the notebook before pointing anyone here, since right now there's no runnable code to show.
