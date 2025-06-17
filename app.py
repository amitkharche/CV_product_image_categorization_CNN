import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="🛍️ Product Image Categorizer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛍️ E-commerce Product Image Categorizer</h1>
    <p>Upload your product images and get instant AI-powered categorization</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and label encoder"""
    try:
        model = tf.keras.models.load_model("model/product_cnn_model.h5")
        with open("model/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        return model, label_encoder, None
    except Exception as e:
        return None, None, str(e)

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def get_confidence_level(max_prob):
    """Determine confidence level based on prediction probability"""
    if max_prob >= 0.8:
        return "High", "confidence-high"
    elif max_prob >= 0.5:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

# Load model
model, label_encoder, error = load_model()

if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("Please ensure the model files are in the correct location:")
    st.code("model/product_cnn_model.h5\nmodel/label_encoder.pkl")
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="upload-section">
        <h3>📤 Upload Product Image</h3>
        <p>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear product image for best results"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image, 
            caption=f"📸 {uploaded_file.name}", 
            use_container_width=True
        )
        
        # Image info
        st.markdown("### 📊 Image Details")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Width", f"{image.size[0]}px")
        with col_info2:
            st.metric("Height", f"{image.size[1]}px")

with col2:
    if uploaded_file:
        with st.spinner("🤖 Analyzing image..."):
            # Preprocess and predict
            input_tensor = preprocess_image(image)
            prediction = model.predict(input_tensor)[0]
            predicted_idx = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
            confidence = prediction[predicted_idx]
            
            # Get confidence level
            conf_level, conf_class = get_confidence_level(confidence)
            
            # Prediction card
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Prediction Result</h2>
                <h1>{predicted_label.upper()}</h1>
                <p class="{conf_class}">Confidence: {conf_level} ({confidence:.1%})</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence metrics
            st.markdown("### 📈 Prediction Confidence")
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            
            with col_conf1:
                st.metric(
                    "Confidence Score", 
                    f"{confidence:.1%}",
                    delta=f"{conf_level} confidence"
                )
            
            with col_conf2:
                st.metric(
                    "Category Rank", 
                    f"#{predicted_idx + 1}",
                    delta="Best match"
                )
            
            with col_conf3:
                total_categories = len(label_encoder.classes_)
                st.metric(
                    "Total Categories", 
                    total_categories
                )
    
    else:
        st.markdown("""
        <div class="stats-card">
            <h3>🚀 How it works</h3>
            <ol>
                <li><strong>Upload</strong> a product image</li>
                <li><strong>AI analyzes</strong> the image features</li>
                <li><strong>Get instant</strong> category prediction</li>
                <li><strong>View confidence</strong> scores and details</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Detailed results section
if uploaded_file:
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")
    
    # Create DataFrame for all predictions
    categories = label_encoder.classes_
    pred_df = pd.DataFrame({
        'Category': categories,
        'Confidence': prediction,
        'Probability': [f"{p:.1%}" for p in prediction]
    }).sort_values('Confidence', ascending=False)
    
    # Top predictions chart
    top_5 = pred_df.head(5)
    fig = px.bar(
        top_5, 
        x='Confidence', 
        y='Category',
        orientation='h',
        title="Top 5 Category Predictions",
        color='Confidence',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # All predictions table
    with st.expander("📋 View All Category Predictions"):
        st.dataframe(
            pred_df,
            use_container_width=True,
            hide_index=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 Powered by TensorFlow & Streamlit | Built for E-commerce Classification</p>
</div>
""", unsafe_allow_html=True)