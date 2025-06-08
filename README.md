# 🛍️ Product Image Categorization

## 📌 Business Use Case

E-commerce platforms handle millions of product listings. Manual categorization of product images is:
- Time-consuming
- Error-prone
- Costly at scale

**Product Image Categorization** automates this by using computer vision to tag images with appropriate categories (e.g., Shoe, Phone, Bag, Watch, Book).

### Applications:
- Smart cataloging
- Improved product discovery
- Better search and recommendations

---

## ⚙️ Features
- CNN classifier trained on simulated product images
- Label encoding for classes
- Prediction via Streamlit UI
- Ready for real-world datasets like Amazon Product Image Dataset

---

## 🧪 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train model
```bash
python model_training.py
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

```bash
docker build -t product-categorizer .
docker run -p 8501:8501 product-categorizer
```

---

## 📁 Structure

```
product_image_categorization_project/
├── data/
│   ├── images/
│   └── product_labels.csv
├── model/
│   └── product_cnn_model.h5
│   └── label_encoder.pkl
├── app.py
├── model_training.py
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .gitattributes
```

## 📜 License

MIT License
