# Multimodal Bioage Prediction

This repository contains the implementation of a multimodal image classification model that uses Swin Transformers for feature extraction from different types of images (e.g., face, tongue, and fundus) and a cross-attention mechanism to integrate features for final classification.

## Structure

The code is organized as follows:

- `dataset.py`: Contains the `MultiModalDataset` class that handles loading and transforming the images.
- `model/swin_transformer.py`: Includes the `SwinTransformerBlock` and `SwinTransformerEncoder` classes for the Swin Transformer model (note: you need to implement the actual transformer block).
- `model/cross_attention.py`: Contains the `CrossAttentionModule` class for the cross-attention mechanism.
- `model/classifier.py`: Contains the `MLPHeader` and `ClassifierModel` classes for the MLP headers and the final classification model.
- `main.py`: The main script to run the training and evaluation of the model.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
To train the model, you need to set up your dataset by specifying the image paths and labels. Then run:

```python
python main.py
```

This will start the training process and save the model after training. You can also use the provided code to perform validation and make predictions.

## License
This project is licensed under the Apache-2.0 license - see the LICENSE file for details.
