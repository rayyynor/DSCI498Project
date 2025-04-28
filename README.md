# DSCI498Project
# HappiScore

## 📌 Project Overview
This project aims to develop an model that **analyzes facial expressions and assigns a smile rating score from 1 to 10**. Unlike traditional emotion classification models that simply categorize expressions as "happy" or "sad," this system provides a **more detailed and quantified evaluation of smiles**. By leveraging **deep learning models**, the system assesses facial features and smile intensity, generating a meaningful score.

The final model is deployed as a **web-based dashboard**, allowing users to **upload images and receive instant smile ratings**. This has practical applications in **entertainment, social media, marketing, and psychology**, enabling **objective measurement of positive emotions** in various domains.

## Project Goals
Develop a model that quantifies happiness score(1-10) based on facial expressions using deep learning and generative models.
- **Detect and analyze** facial expressions to measure smile intensity.
- **Develop a regression-based deep learning model** that assigns a numerical smile rating.
- **Deploy a web-based platform** where users can upload images and receive an instant smile score.

## Why This Matters
A system that **rates smiles** rather than just detecting happiness has several valuable applications:
- **Social Media & Entertainment** – Helps users analyze their smiles before posting photos or measure engagement in online interactions.
- **Marketing & Customer Experience** – Brands can track audience reactions and quantify positive expressions in advertisements or live events.
- **Photography & Events** – Photographers can gauge whether people are genuinely smiling in group pictures or candid shots.
- **Psychological & Behavioral Studies** – Researchers can use this system to study emotional intensity in facial expressions and human behavior.

By introducing a **numerical scale for smiles**, we move beyond simple "happy or not" classifications and enable a **data-driven approach to measuring positive emotions**.

## Dataset Overview
For this project, we utilized the **FER2013 (Facial Expression Recognition 2013) dataset**, a widely used facial emotion dataset containing **35,000+ grayscale facial images** categorized into various emotion labels. Since FER2013 only classifies emotions into discrete categories, we transformed this data into a **continuous smile rating scale (1-10)** to achieve a more precise assessment.

### Data Processing Steps
1️⃣ **Face Detection & Cropping** – Used **MTCNN (Multi-task Cascaded Convolutional Networks)** to detect and crop faces, ensuring the model focuses on relevant facial features.  
2️⃣ **Feature Extraction** – Leveraged **ResNet50**, a pre-trained deep learning model, to extract high-level facial features and analyze smile intensity.  
3️⃣ **Smile Intensity Mapping** – Converted emotion labels into a **continuous smile rating score from 1 to 10**, allowing for a more refined analysis of smile intensity.  
4️⃣ **Data Augmentation** – Used **GANs (Generative Adversarial Networks)** to generate additional smiling faces with varying intensities, improving dataset diversity and model performance.  

##  Model Architecture & Training
The model is built using **PyTorch**, leveraging **ResNet50** as the backbone for feature extraction. The **final output layer is modified** to produce a **continuous score (1-10) instead of class labels**.

### ** Model Training Process:**
1. **Input:** Preprocessed facial images (224x224 pixels).
2. **Feature Extraction:** ResNet50 extracts deep facial features.
3. **Regression Layer:** The final fully connected layer outputs a single numerical value (smile rating).
4. **Loss Function:** Mean Squared Error (MSE) loss is used to minimize prediction error.
5. **Optimization:** Adam optimizer with a learning rate of 0.001.
6. **Training:** Model trained on GPU using PyTorch, with data augmentation applied to enhance generalization.

## Deployment & Usage
The model is deployed as a **user-friendly web app** using **Streamlit**, allowing real-time smile rating analysis.


# Happyscore  –  Liang Sisheng, Siyuan Cao
Predict how happy a face looks (score 1-10) and generate a happier version.

## Folder layout
├── app.py # Streamlit UI
├── cli_helper.py # quick wrapper (download / train)
├── data_utils.py
├── generator.py # xformers guarded
├── inference.py # imports torch + get_scoring_model
├── model.py
├── train.py # num_workers=0, Grayscale→RGB
└── requirements.txt # clean list, numpy<2 pinned

## Quick start (macOS)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login             # first-time only
python train.py --download-data   # fetch FER-2013
python train.py                   # train scorer (≈12 min on M2)
streamlit run main.py             # set up dashboard on your local


##Configure secrets once per machine
# Kaggle (dataset)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/ # or set KAGGLE_USERNAME / KAGGLE_KEY
chmod 600 ~/.kaggle/kaggle.json
# (Stable-Diffusion)
huggingface-cli login # paste hf_....tc token

#Download FER-2013(if needed)
python cli_helper.py --download-data
