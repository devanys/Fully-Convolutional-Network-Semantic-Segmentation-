# Fully Convolutional Network (Semantic Segmentation)
<img width="1425" height="484" alt="image" src="https://github.com/user-attachments/assets/9a2533ed-6660-4a2a-9022-c3e81a7b1584" />
<img width="1285" height="622" alt="image" src="https://github.com/user-attachments/assets/ab3a1827-2c2b-4003-8e74-5039547548de" />

## Overview

Demonstrates the implementation of a **Fully Convolutional Network (FCN)** for semantic segmentation using PyTorch. The notebook explains the complete workflow of semantic segmentation, including dataset generation, custom loss function implementation, model training, evaluation metrics computation, and visualization of segmentation results.

The notebook is organized in a step-by-step educational structure where each program cell explains a specific stage of the semantic segmentation pipeline.

---

# Fully Convolutional Network (FCN)

A Fully Convolutional Network replaces fully connected layers with convolutional layers so that the network can generate dense pixel-wise predictions.

Unlike traditional image classification models, FCN preserves spatial information and produces segmentation masks corresponding to the input image dimensions.

---

# Notebook Workflow Explanation

## 1. Library Import Cell

This section imports the required libraries such as PyTorch, Torchvision, NumPy, Matplotlib, PIL, and Scikit-learn.

These libraries are used for:

* Deep learning operations
* Image preprocessing
* Visualization
* Model evaluation
* Segmentation inference

---

## 2. Dice Loss Implementation Cell

This cell defines the custom Dice Loss function used for semantic segmentation tasks.

Dice Loss measures overlap between predicted segmentation masks and ground truth masks.

---

## 3. Synthetic Dataset Generation Cell

This section generates artificial segmentation datasets consisting of images and segmentation masks.

The generated dataset is used for training and testing the FCN model.

---

## 4. FCN Model Definition Cell

This cell defines the Fully Convolutional Network architecture.

The architecture contains:

* Convolution layers
* ReLU activations
* Pooling layers
* Upsampling layers
* Pixel-wise classification output

---

## 5. Training Loop Cell

This section performs:

* Forward propagation
* Dice Loss computation
* Backpropagation
* Optimizer update
* Epoch-wise monitoring

The objective is to minimize segmentation error.

---

## 6. Evaluation Metrics Cell

This section computes:

* Dice Score
* IoU (Intersection over Union)
* F1-Score
* Confusion Matrix

These metrics evaluate segmentation performance at the pixel level.

---

## 7. Visualization Cell

This section visualizes:

* Original images
* Ground truth masks
* Predicted masks
* Overlay segmentation results
* Probability heatmaps

---

## 8. DeepLabV3 Inference Cell

This section demonstrates semantic segmentation inference using the pretrained DeepLabV3 model from Torchvision.

---

# Mathematical Formulation

## 1. Convolution Operation

$$
Y(i,j) = \sum_m \sum_n X(i-m,j-n)K(m,n)
$$

Where:

* $X$ = input feature map
* $K$ = convolution kernel
* $Y$ = output feature map

---

## 2. ReLU Activation Function

$$
f(x) = \max(0,x)
$$

The ReLU activation introduces non-linearity into the network.

---

## 3. Pooling Operation

$$
Y(i,j) = \max_{(m,n) \in R} X(m,n)
$$

Where:

* $R$ = pooling region

---

## 4. Upsampling Operation

$$
Y = \text{Upsample}(X)
$$

Upsampling restores spatial resolution for dense segmentation output.

---

## 5. Softmax Probability

$$
P(y_i)=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
$$

Where:

* $z_i$ = output logits
* $C$ = number of segmentation classes

---

## 6. Dice Coefficient

$$
Dice = \frac{2|X \cap Y|}{|X|+|Y|}
$$

This metric measures overlap between prediction and ground truth segmentation masks.

---

## 7. Dice Loss

$$
\mathcal{L}_{Dice}=1-\frac{2\sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}
$$

Where:

* $p_i$ = predicted probability
* $g_i$ = ground truth mask
* $\epsilon$ = smoothing constant

---

## 8. Intersection over Union (IoU)

$$
IoU = \frac{TP}{TP+FP+FN}
$$

Where:

* $TP$ = True Positive
* $FP$ = False Positive
* $FN$ = False Negative

---

## 9. Cross Entropy Loss

$$
\mathcal{L}*{CE} = -\sum*{i=1}^{C} y_i \log(\hat{y}_i)
$$

Where:

* $y_i$ = ground truth label
* $\hat{y}_i$ = predicted probability

---

# DeepLabV3 Architecture

This project also demonstrates semantic segmentation inference using the pretrained **DeepLabV3-ResNet50** model from Torchvision.

DeepLabV3 utilizes:

* Atrous (dilated) convolution
* Multi-scale feature extraction
* Residual learning
* Spatial pyramid pooling

to improve segmentation accuracy.

---

# Results

## FCN Visualization

<img width="1529" height="565" alt="image" src="https://github.com/user-attachments/assets/7051cd42-3b78-412d-aaee-9ec44b2e221d" />

---

## Training Curve

<img width="1541" height="492" alt="image" src="https://github.com/user-attachments/assets/37cbec0c-3eaf-4e02-977a-069a219c86ad" />


## Confusion Matrix

<img width="1522" height="572" alt="image" src="https://github.com/user-attachments/assets/e2dd60f3-44ef-4a5b-b4b5-e747dc15277c" />

---

## Segmentation Overlay Result

<img width="959" height="781" alt="image" src="https://github.com/user-attachments/assets/e2d5b296-2104-49d6-bd10-8972450e5d42" />
<img width="1550" height="555" alt="image" src="https://github.com/user-attachments/assets/74ad6bf7-3140-4127-ae6c-12c9b6af4a5a" />
---

## Probability Heatmaps

<img width="1424" height="649" alt="image" src="https://github.com/user-attachments/assets/6ff641bd-0750-4019-b2a7-ab9ae782c553" />

---

# References

## Papers

1. Long, J., Shelhamer, E., & Darrell, T. (2015). *Fully Convolutional Networks for Semantic Segmentation*. CVPR.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.

3. Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). *Rethinking Atrous Convolution for Semantic Image Segmentation*.

---
