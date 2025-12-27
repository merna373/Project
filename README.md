# Chest X-Ray Images with Pneumothorax Masks for Semantic Segmentation

## Project Overview
This project is a **deep learning system** for **semantic segmentation** of **pneumothorax** (collapsed lung) in chest X-ray images.  
The system uses a **Swin Transformer-based U-Net architecture** to accurately detect and segment pneumothorax regions, helping radiologists in diagnosis.

---

## Key Features
- **Semantic Segmentation:** Pixel-level detection of pneumothorax regions  
- **Advanced Architecture:** Swin Transformer encoder + U-Net decoder  
- **Data Augmentation:** Specialized transformations for medical imaging  
- **Comprehensive Evaluation:** Dice score, IoU, and visualization tools  
- **Web Interface:** User-friendly GUI for clinical use  

---

## Dataset
- **Chest X-Ray Images with Pneumothorax Masks**  
- Dataset from the **SIIM-ACR Pneumothorax Segmentation Challenge**

### Data Description
- **Number of Images:** 12,047 chest X-ray images  
- **Number of Masks:** 12,047 corresponding pneumothorax segmentation masks  
- **Image Size:** 1024 × 1024  
- **Image Format:** PNG  
- **Mask Type:** Binary segmentation masks  
  - `0` → No pneumothorax  
  - `1` → Pneumothorax present  

Each image has a corresponding mask with the same filename, stored in a separate folder.

---

## Project Structure
```text
Pneumothorax-Detection/
├── data/
│   ├── png_images/
│   ├── png_masks/
│   └── splits/
│
├── notebooks/
│   └── DeepLearning_Project.ipynb
│
├── models/
│   └── best_swin_unet.pth
│
├── web_app/
│   ├── frontend/
│   └── backend/
│
├── requirements.txt
└── README.md
```
## Data Preprocessing Pipeline
- **Data Validation:** Check for missing/corrupted images  
- **Class Balance Analysis:** 78% normal vs 22% pneumothorax  
- **Mask Normalization:** Convert to binary format (0/255)  
- **Quality Assessment:** Analyze brightness, contrast, noise  
- **Dataset Splitting:** Train/Val/Test (balanced validation)

### Data Augmentation
- Vertical flipping  
- Random brightness/contrast  
- Gaussian noise  
- CLAHE enhancement  
- Random affine transformations  

---

## Model Architecture
- **Encoder:** Swin-Tiny Transformer (pretrained on ImageNet)  
- **Decoder:** U-Net style with skip connections  
- **Loss Function:** Combined BCE + Dice Loss  
- **Metrics:** Dice Score, IoU  


## Model Details
### SwinUNet Architecture
```
Input (1, 224, 224)
→ Swin-Tiny Encoder
→ UNet Decoder
→ Output (1, 224, 224)
```
- **Encoder Output Channels:** `[96, 192, 384, 768]`  
- **Decoder:** 5 upsampling blocks with skip connections  
- **Final Layer:** 1×1 convolution + sigmoid activation  


## Training Configuration
- **Loss Function:** BCE + Dice Loss (α = 0.5)  
- **Optimizer:** Adam (lr = 1e-4)  
- **Batch Size:** 8  
- **Epochs:** 50 (with early stopping)  
- **Evaluation Metric:** Dice Score 

---

Technologies Used
Backend (Python)

Frontend (Web Interface)



---

## Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pneumothorax-detection.git
cd pneumothorax-detection
```
### 2. Install dependencies:
#### Make sure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```
### 3. Prepare the Dataset
#### Download the SIIM-ACR Pneumothorax Segmentation dataset.
#### Place the data in the following structure:
```bash
data/
├── png_images/
├── png_masks/
```

### 4. Run the Project (Jupyter Notebook)
#### All preprocessing, visualization, augmentation, model training, evaluation, and inference are implemented inside a single notebook.

### Option 1: Using Jupyter Notebook
```bash
jupyter notebook
```
### Open:
```bash
notebooks/DeepLearning_Project.ipynb
```
#### Run all cells sequentially from top to bottom.

### Option 2: Using JupyterLab
```bash
jupyter lab
```
#### Then open the notebook and run all cells.

### 5. Run Inference on New Images
#### Place new X-ray images in the specified inference folder inside the notebook.

#### Load the trained model:
```bash
models/best_swin_unet.pth
```
#### Run the inference cells to generate predicted masks.
---
## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Best Validation Dice Score | 0.85+ |
| Test Dice Score | 0.83+ |
| IoU | 0.75+ |
| Training Time | ~4 hours on GPU |

### Sample Predictions
![Sample Predictions]()

---
## Team Roles

| Team Member | Responsibilities |
|------------|-----------------|
| Nourhan    | Frontend website development |
| Zyad       | Backend server development |
| Shrouk     | Data preprocessing & analysis |
| Hadeer & Howaida | Data visualization, augmentation, DataLoader |
| Yasmine    | Model design & loss functions |
| Merna      | Model training, evaluation & visualization |

