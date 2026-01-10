#  Real-Time Forest Fire Detection using FireNet-CNN and Explainable AI

This project implements **FireNet-CNN**, a lightweight convolutional neural network for real-time forest fire detection from images. The model classifies images into **Fire** and **No-Fire** categories and uses **Explainable AI (Grad-CAM)** to visualize and interpret model decisions.

This work is a complete end-to-end reproduction of a published research paper using publicly available datasets.

---

##  Project Highlights
- Custom lightweight CNN architecture (FireNet-CNN)
- Binary image classification (Fire / No-Fire)
- High recall suitable for safety-critical applications
- Explainable AI using Grad-CAM
- Confusion matrix and failure case analysis
- Fully reproducible pipeline
- Suitable for real-time and edge deployment

---

## Dataset Information

The original research paper does not provide a downloadable dataset. Therefore, publicly available forest fire image datasets were used.

### Dataset Summary
| Category | Images |
|--------|--------|
| Fire | 950 |
| No-Fire | 950 |
| **Total** | **1,900** |

### Dataset Split
| Split | Fire | No-Fire | Total |
|-----|------|--------|-------|
| Training | 608 | 608 | 1216 |
| Validation | 152 | 152 | 304 |
| Testing | 190 | 190 | 380 |

✔ Balanced dataset  
✔ No data leakage  
✔ Test set kept completely unseen during training  

---

##  Model Architecture (FireNet-CNN)

- 5 Convolutional Blocks
- Filters: `32 → 64 → 128 → 256 → 512`
- Batch Normalization + ReLU
- Max Pooling
- Fully Connected layers with Dropout
- Binary output (Fire / No-Fire)

**Total Parameters:** ~2.7 Million  
**Model Type:** Lightweight CNN (no pretrained weights)

---

## Training Configuration

- Optimizer: Adam  
- Learning Rate: `1e-4`  
- Loss Function: Binary Cross-Entropy with Logits  
- Batch Size: 32  
- Epochs: 40  
- Image Size: `150 × 150`

---

##  Test Set Results

| Metric | Value |
|------|-------|
| Accuracy | 95.53% |
| Precision | 93.91% |
| Recall | 97.37% |
| F1 Score | 95.61% |

High recall ensures minimal missed fire detections, which is critical for wildfire monitoring systems.

---

##  Explainable AI (Grad-CAM)

Grad-CAM is applied to the final convolutional layer of FireNet-CNN to visualize important regions influencing predictions.

### Explainability Cases Generated
- True Positive (Fire → Fire)
- True Negative (No Fire → No Fire)
- False Positive (No Fire → Fire)
- False Negative (Fire → No Fire)

Grad-CAM visualizations confirm that the model focuses on flame and smoke regions, while misclassifications occur in visually ambiguous scenarios such as fog, sunlight, or low-intensity fire.

---

---


---

##  Technologies Used
- Python
- PyTorch
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- Scikit-learn

---
## Models Used
 - FireNet-CNN (Proposed) : Custom CNN (5 convolution blocks) , ~2.7M parameters ,Designed specifically for fire detection

 - VGG-16 (Baseline) : Pretrained ImageNet model , ~138M parameters , Used only for comparison

📈 Test Performance
| Model       | Accuracy   | Precision  | Recall     | F1-score   |
| ----------- | ---------- | ---------- | ---------- | ---------- |
| FireNet-CNN | **95.53%** | **93.91%** | **97.37%** | **95.61%** |
| VGG-16      | 94.00%     | 92.38%     | 95.79%     | 94.06%     |


---
## Confusion Matrix

## VGG-16

<img width="378" height="393" alt="image" src="https://github.com/user-attachments/assets/62db1e1c-838d-4c82-a47b-fd76f0ce4a60" />





## FireNet-CNN

<img width="444" height="393" alt="image" src="https://github.com/user-attachments/assets/24458de1-7a13-46e2-9bde-365191cf788c" />




## Future Work
- Synthetic data generation using diffusion models
- Video-based forest fire detection
- Fire severity classification
- Edge device deployment (Jetson / Raspberry Pi)
- Multi-sensor fusion (vision + thermal)

---

## Reference
Real-Time Detection of Forest Fires Using FireNet-CNN and Explainable AI Techniques, IEEE Access.(https://ieeexplore.ieee.org/document/10930496/)
