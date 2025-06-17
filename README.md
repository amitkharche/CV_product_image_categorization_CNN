
---

# 🛍️ Product Image Categorization

## 📌 Business Use Case

E-commerce platforms handle millions of product listings. Manual categorization of product images is:

* Time-consuming
* Error-prone
* Costly at scale

**Product Image Categorization** automates this by using computer vision to tag images with appropriate product categories (e.g., **Shoe, Phone, Bag, Watch, Book**).

### ✅ Applications

* Smart cataloging
* Improved product discovery
* Enhanced search and recommendation systems

---

## ⚙️ Features

* CNN-based classifier trained on simulated product images
* Label encoding for product categories
* Stylish **Streamlit UI** for image prediction
* Modular codebase, ready to scale on real-world datasets

---

## 🧪 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/amitkharche/CV_product_image_categorization_CNN.git
cd CV_product_image_categorization_CNN
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python model_training.py
```

This will:

* Load and preprocess product images from `data/`
* Encode category labels
* Train a CNN image classifier
* Save the model (`product_cnn_model.h5`) and label encoder

### 4. Launch the web app

```bash
streamlit run app.py
```

---

## 🐳 Run with Docker (Optional)

```bash
docker build -t product-categorizer .
docker run -p 8501:8501 product-categorizer
```

---

## 📁 Project Structure

```
product_image_categorization_project/
├── data/
│   ├── images/                    # Product images
│   └── product_labels.csv         # CSV with image paths & categories
├── model/
│   ├── product_cnn_model.h5       # Trained CNN model
│   └── label_encoder.pkl          # Encoded label mappings
├── app.py                         # Streamlit web app
├── model_training.py              # CNN training script
├── requirements.txt               # Dependencies
├── Dockerfile                     # For Docker builds
├── .gitignore
├── .gitattributes
└── README.md                      # You're here!
```

---

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute with attribution.

---

## 🤝 Let’s Connect!

Have questions or want to collaborate?

* 💼 [LinkedIn](https://www.linkedin.com/in/amit-kharche)
* 📝 [Medium](https://medium.com/@amitkharche14)
* 💻 [GitHub](https://github.com/amitkharche)

---
