import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import cv2
import requests
import os
from pathlib import Path
import gdown

# Set page config
st.set_page_config(
    page_title="♻️ EcoSense",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with gradient backgrounds and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .hero-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .hero-description {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.8;
    }
    
    .modern-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(17, 153, 142, 0.3);
    }
    
    .prediction-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .confidence-score {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1rem 0;
        background: rgba(255,255,255,0.2);
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #00f2fe;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #fee140;
        box-shadow: 0 8px 25px rgba(250, 112, 154, 0.3);
    }
    
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    }
    
    .class-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        color: white;
        font-weight: 600;
        margin: 0.25rem;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .progress-container {
        background: rgba(0,0,0,0.1);
        border-radius: 10px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.8s ease;
    }
    
    .sidebar-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .data-insight {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 8px 25px rgba(168, 237, 234, 0.3);
    }
    
    .footer-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prob_df' not in st.session_state:
    st.session_state.prob_df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def download_model_from_gdrive():
    """Download model from Google Drive using multiple methods."""
    
    # Google Drive file ID extracted from your link
    file_id = "1gMSisnKZeUxFovfI6t2Is-V77v0Jj03s"
    model_path = "sequential.h5"
    
    # Check if model already exists
    if os.path.exists(model_path):
        return model_path
    
    try:
        # Method 1: Using gdown (recommended)
        st.info("📥 Downloading model from Google Drive... (first time only)")
        progress_bar = st.progress(0)
        
        # Download using gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
        
        progress_bar.progress(100)
        
        if os.path.exists(model_path):
            st.success(f"✅ Model downloaded successfully: {model_path}")
            return model_path
        else:
            raise Exception("Download completed but file not found")
            
    except Exception as e:
        st.error(f"Error with gdown method: {e}")
        
        try:
            # Method 2: Direct download with requests
            st.info("🔄 Trying alternative download method...")
            
            # Use direct download URL
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = requests.get(download_url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = int((downloaded / total_size) * 100)
                                progress_bar.progress(progress)
                
                progress_bar.progress(100)
                st.success(f"✅ Model downloaded successfully: {model_path}")
                return model_path
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e2:
            st.error(f"Error with requests method: {e2}")
            
            # Method 3: Manual instruction
            st.markdown("""
            <div class="status-warning">
                ⚠️ <strong>Automatic download failed.</strong><br>
                Please download the model manually:
                <ol>
                    <li>Go to: <a href="https://drive.google.com/file/d/1gMSisnKZeUxFovfI6t2Is-V77v0Jj03s/view?usp=sharing" target="_blank">Google Drive Link</a></li>
                    <li>Click "Download"</li>
                    <li>Save the file as "sequential.h5" in the same folder as this script</li>
                    <li>Refresh this page</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            return None

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the trained model from Google Drive."""
    try:
        # First, try to download the model
        model_path = download_model_from_gdrive()
        
        if model_path and os.path.exists(model_path):
            # Load the actual trained model
            st.info("🔄 Loading trained model...")
            model = tf.keras.models.load_model(model_path)
            
            # Display model architecture info
            st.markdown(f"""
            <div class="status-success">
                ✅ <strong>Success!</strong> Your trained model has been loaded successfully.<br>
                <strong>Model Summary:</strong><br>
                - Total parameters: {model.count_params():,}<br>
                - Input shape: {model.input_shape}<br>
                - Output classes: {model.output_shape[-1]}<br>
                - Expected input size: 256×256×3 (as per notebook)
            </div>
            """, unsafe_allow_html=True)
            
            return model
        else:
            st.warning("⚠️ Model file not available. Please ensure sequential.h5 is downloaded.")
            return None
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please check if the model file is compatible and properly trained.")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction - EXACTLY matching notebook preprocessing."""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Ensure RGB format (notebook uses RGB)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Already RGB, good to go
            pass
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA, convert to RGB
            img_array = img_array[:, :, :3]
        
        # Resize to model input size (256×256 as per notebook - NOT 224×224)
        img_array = cv2.resize(img_array, (256, 256))
        
        # Normalize pixel values to [0, 1] as per notebook (rescale=1.0/255.0)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image_array):
    """Make prediction on the preprocessed image."""
    try:
        if model is None:
            st.error("Model not loaded. Cannot make predictions.")
            return None
            
        if image_array is None:
            st.error("Image preprocessing failed. Cannot make predictions.")
            return None
        
        # Make actual prediction
        st.info(f"Making prediction with input shape: {image_array.shape}")
        predictions = model.predict(image_array, verbose=0)
        
        # Ensure we have the right number of classes (5 classes as per notebook)
        if len(predictions[0]) == 5:
            return predictions[0]
        else:
            st.warning(f"Model output has {len(predictions[0])} classes, expected 5")
            return predictions[0]
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error("This usually indicates an architecture mismatch between the saved model and expected input.")
        
        # Provide debugging information
        if model is not None:
            st.error(f"Model input shape: {model.input_shape}")
            st.error(f"Model output shape: {model.output_shape}")
            
        if image_array is not None:
            st.error(f"Input array shape: {image_array.shape}")
            
        return None

# Class information with modern colors
CLASS_INFO = {
    'cardboard': {
        'original_count': 393, 
        'final_count': 314,
        'color': '#FF6B6B', 
        'description': 'Recyclable cardboard materials including boxes and packaging',
        'icon': '📦',
        'tips': 'Remove tape and flatten before recycling'
    },
    'glass': {
        'original_count': 491, 
        'final_count': 392,
        'color': '#4ECDC4', 
        'description': 'Glass bottles, jars, and containers',
        'icon': '🍶',
        'tips': 'Rinse clean and remove caps/lids'
    },
    'metal': {
        'original_count': 400, 
        'final_count': 320,
        'color': '#95A5A6', 
        'description': 'Metal cans, aluminum containers, and metallic objects',
        'icon': '🥫',
        'tips': 'Clean containers and remove labels'
    },
    'paper': {
        'original_count': 584, 
        'final_count': 467,
        'color': '#F39C12', 
        'description': 'Paper documents, newspapers, and paper materials',
        'icon': '📄',
        'tips': 'Keep dry and remove plastic coatings'
    },
    'plastic': {
        'original_count': 472, 
        'final_count': 377,
        'color': '#9B59B6', 
        'description': 'Plastic bottles, containers, and plastic waste',
        'icon': '🍼',
        'tips': 'Check recycling code and clean thoroughly'
    }
}

# Class names exactly as in notebook
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

def create_modern_sidebar():
    """Create modern sidebar with enhanced visuals."""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📊 EcoSense Dashboard</div>', unsafe_allow_html=True)
        
        # Model status
        model_status = "sequential.h5" if os.path.exists("sequential.h5") else "not found"
        
        if model_status == "sequential.h5":
            st.markdown("""
            <div class="status-success">
                ✅ <strong>Model Ready</strong><br>
                Your trained model is loaded and ready for classification!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-warning">
                ⚠️ <strong>Model Loading</strong><br>
                Will download from Google Drive on first use.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset overview with modern cards
        st.markdown("### 🎯 Material Classes")
        
        total_original = sum(info['original_count'] for info in CLASS_INFO.values())
        total_final = sum(info['final_count'] for info in CLASS_INFO.values())
        
        for class_name, info in CLASS_INFO.items():
            final_pct = (info['final_count'] / total_final) * 100
            
            st.markdown(f"""
            <div class="modern-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{info['icon']}</span>
                    <strong style="color: {info['color']}; font-size: 1.1rem;">{class_name.title()}</strong>
                </div>
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">
                    {info['description']}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                    <span><strong>{info['final_count']}</strong> training images</span>
                    <span><strong>{final_pct:.1f}%</strong> of dataset</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {final_pct}%; background: {info['color']};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model architecture info
        st.markdown("### 🧠 Model Info")
        st.markdown("""
        <div class="data-insight">
            <h4 style="margin-top: 0; color: #667eea;">🔬 CNN Architecture</h4>
            <ul style="margin: 0;">
                <li><strong>Input:</strong> 256×256×3 RGB images</li>
                <li><strong>Layers:</strong> 3 Conv2D blocks + Dense layers</li>
                <li><strong>Parameters:</strong> Optimized for accuracy</li>
                <li><strong>Training:</strong> 80/20 split with class balancing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("### 📈 Training Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_final:,}</div>
                <div class="metric-label">Training Images</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-value">5</div>
                <div class="metric-label">Material Classes</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Controls
        if st.button("🔄 Reload Model", help="Re-download and reload the model"):
            if os.path.exists("sequential.h5"):
                os.remove("sequential.h5")
            st.cache_resource.clear()
            st.rerun()
        
        debug_mode = st.checkbox("🔍 Debug Mode", help="Show detailed model architecture")
        return debug_mode

def create_prediction_display(predicted_class, confidence, predictions):
    """Create modern prediction result display."""
    class_info = CLASS_INFO[predicted_class]
    
    # Main prediction result
    st.markdown(f"""
    <div class="prediction-result animate-fade-in">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{class_info['icon']}</div>
        <div class="prediction-title">{predicted_class.upper()}</div>
        <div class="confidence-score">{confidence:.1f}% Confidence</div>
        <div style="font-size: 1.1rem; margin-top: 1rem; opacity: 0.9;">
            {class_info['description']}
        </div>
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px;">
            <strong>♻️ Recycling Tip:</strong> {class_info['tips']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed breakdown
    st.markdown("### 📊 Classification Breakdown")
    
    # Create probability dataframe
    prob_df = pd.DataFrame({
        'Material': [CLASS_NAMES[i].title() for i in range(len(predictions))],
        'Probability': predictions * 100,
        'Color': [CLASS_INFO[CLASS_NAMES[i]]['color'] for i in range(len(predictions))],
        'Icon': [CLASS_INFO[CLASS_NAMES[i]]['icon'] for i in range(len(predictions))]
    }).sort_values('Probability', ascending=False)
    
    # Display top 3 predictions as cards
    col1, col2, col3 = st.columns(3)
    
    for idx, (col, (_, row)) in enumerate(zip([col1, col2, col3], prob_df.head(3).iterrows())):
        with col:
            rank_style = "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)" if idx == 0 else \
                        "linear-gradient(135deg, #C0C0C0 0%, #A0A0A0 100%)" if idx == 1 else \
                        "linear-gradient(135deg, #CD7F32 0%, #A0522D 100%)"
            
            st.markdown(f"""
            <div class="modern-card" style="background: {rank_style}; color: white; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{row['Icon']}</div>
                <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                    #{idx + 1} {row['Material']}
                </div>
                <div style="font-size: 1.5rem; font-weight: 700;">
                    {row['Probability']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    return prob_df

def create_probability_charts(prob_df):
    """Create modern probability visualization charts with error handling."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Horizontal bar chart with simplified configuration
            fig_bar = go.Figure(data=[
                go.Bar(
                    y=prob_df['Material'],
                    x=prob_df['Probability'],
                    orientation='h',
                    marker=dict(
                        color=prob_df['Color'].tolist(),
                        line=dict(color='rgba(255,255,255,0.6)', width=2)
                    ),
                    text=[f"{icon} {prob:.1f}%" for icon, prob in zip(prob_df['Icon'], prob_df['Probability'])],
                    textposition='inside'
                )
            ])
            
            fig_bar.update_layout(
                title="🎯 Probability Rankings",
                xaxis_title="Confidence (%)",
                yaxis_title="Material Type",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Arial', size=12),
                height=350
            )
            
            fig_bar.update_xaxes(range=[0, 100], showgrid=True, gridcolor='rgba(0,0,0,0.1)')
            fig_bar.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
        except Exception as e:
            st.error("Error creating bar chart")
            # Fallback to simple display
            for _, row in prob_df.iterrows():
                st.write(f"{row['Icon']} {row['Material']}: {row['Probability']:.1f}%")
    
    with col2:
        try:
            # Modern donut chart with simplified configuration
            fig_donut = go.Figure(data=[go.Pie(
                labels=[f"{icon} {material}" for icon, material in zip(prob_df['Icon'], prob_df['Material'])],
                values=prob_df['Probability'],
                hole=.6,
                marker=dict(
                    colors=prob_df['Color'].tolist(),
                    line=dict(color='white', width=3)
                ),
                textinfo='percent',
                textposition='outside'
            )])
            
            fig_donut.update_layout(
                title="🍰 Confidence Distribution",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Arial', size=12),
                height=350,
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{prob_df.iloc[0]['Material']}</b><br>{prob_df.iloc[0]['Probability']:.1f}%",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )]
            )
            
            st.plotly_chart(fig_donut, use_container_width=True)
            
        except Exception as e:
            st.error("Error creating donut chart")
            # Fallback to metric display
            st.metric(
                label=f"{prob_df.iloc[0]['Icon']} Top Prediction",
                value=f"{prob_df.iloc[0]['Material']}",
                delta=f"{prob_df.iloc[0]['Probability']:.1f}%"
            )

def create_dataset_overview():
    """Create modern dataset overview section with fixed Plotly configuration."""
    st.markdown("### 📊 Training Dataset Overview")
    
    # Create dataset visualization
    dataset_data = []
    for cls, info in CLASS_INFO.items():
        dataset_data.append({
            'Material': cls.title(),
            'Original': info['original_count'],
            'Training': info['final_count'],
            'Color': info['color'],
            'Icon': info['icon']
        })
    
    dataset_df = pd.DataFrame(dataset_data)
    
    try:
        # Modern grouped bar chart with simplified configuration
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Original Dataset',
            x=dataset_df['Material'],
            y=dataset_df['Original'],
            marker_color='rgba(102, 126, 234, 0.7)',
            text=dataset_df['Original'],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Training Set',
            x=dataset_df['Material'],
            y=dataset_df['Training'],
            marker_color=dataset_df['Color'].tolist(),
            text=dataset_df['Training'],
            textposition='outside'
        ))
        
        # Simplified layout configuration to avoid errors
        fig.update_layout(
            title="📈 Dataset Distribution: Original vs Training Split",
            xaxis_title="Material Types",
            yaxis_title="Number of Images",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            barmode='group',
            height=400,
            font=dict(family='Arial', size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axis properties separately to avoid conflicts
        fig.update_xaxes(
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        )
        
        fig.update_yaxes(
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        # Fallback to simple table if chart fails
        st.dataframe(dataset_df[['Material', 'Original', 'Training']], use_container_width=True)
    
    # Dataset insights cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_original = sum(info['original_count'] for info in CLASS_INFO.values())
    total_training = sum(info['final_count'] for info in CLASS_INFO.values())
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_original:,}</div>
            <div class="metric-label">Original Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <div class="metric-value">{total_training:,}</div>
            <div class="metric-label">Training Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="metric-value">80/20</div>
            <div class="metric-label">Train/Test Split</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">5</div>
            <div class="metric-label">Material Classes</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function with modern design and improved error handling."""
    
    # Hero Section
    st.markdown("""
    <div class="hero-header animate-fade-in">
        <div class="hero-title">♻️ EcoSense</div>
        <div class="hero-subtitle">Smart Recyclable Materials Classification</div>
        <div class="hero-description">
            Powered by Deep Learning • Trained on 2,000+ Images • 5 Material Categories
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create modern sidebar
    debug_mode = create_modern_sidebar()
    
    # Main content area
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Upload section
    st.markdown("## 📤 Upload & Classify")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-area">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📸 Select Material Image</h3>
            <p style="margin-bottom: 1rem; color: #666;">
                Upload a clear image of recyclable materials for classification
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG | Optimal size: 256x256 pixels",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image with modern styling
                image = Image.open(uploaded_file)
                
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.image(image, caption="📷 Uploaded Image", use_column_width=True)
                
                # Image metadata
                file_size = len(uploaded_file.getvalue()) / 1024  # KB
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-top: 1rem; padding: 0.5rem; background: #f8f9fa; border-radius: 8px;">
                    <span><strong>Size:</strong> {image.size[0]}×{image.size[1]} px</span>
                    <span><strong>Format:</strong> {image.format or 'Unknown'}</span>
                    <span><strong>File Size:</strong> {file_size:.1f} KB</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Processing info
                st.info("🔄 Image will be automatically resized to 256×256 pixels and normalized for optimal processing")
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                uploaded_file = None
    
    with col2:
        if uploaded_file is not None:
            try:
                # Load model and make prediction
                with st.spinner("🧠 Loading model..."):
                    model = load_model()
                
                if model is not None:
                    # Show model debug info if enabled
                    if debug_mode:
                        with st.expander("🔍 Model Architecture Details"):
                            try:
                                st.code(f"""
Model Input Shape: {model.input_shape}
Model Output Shape: {model.output_shape}
Total Parameters: {model.count_params():,}
Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}
""")
                            except Exception as debug_error:
                                st.error(f"Debug info error: {debug_error}")
                    
                    # Make prediction
                    with st.spinner("🤖 Analyzing image with AI..."):
                        processed_image = preprocess_image(image)
                        predictions = predict_image(model, processed_image)
                    
                    if predictions is not None:
                        # Ensure we have the right number of predictions
                        if len(predictions) >= len(CLASS_NAMES):
                            predictions = predictions[:len(CLASS_NAMES)]
                        
                        # Get top prediction
                        predicted_class_idx = np.argmax(predictions)
                        predicted_class = CLASS_NAMES[predicted_class_idx]
                        confidence = predictions[predicted_class_idx] * 100
                        
                        # Display prediction result
                        prob_df = create_prediction_display(predicted_class, confidence, predictions)
                        
                        # Store results for later use
                        st.session_state['prediction_made'] = True
                        st.session_state['prob_df'] = prob_df
                        st.session_state['predictions'] = predictions
                else:
                    st.error("❌ Could not load the model. Please check the model file.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            # Welcome message when no image is uploaded
            st.markdown("""
            <div class="modern-card" style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
                <h3 style="color: #667eea; margin-bottom: 1rem;">Ready for Classification</h3>
                <p style="color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
                    Upload an image to get started with automatic recyclable material classification
                </p>
                <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                    <span class="class-badge" style="background: #FF6B6B;">📦 Cardboard</span>
                    <span class="class-badge" style="background: #4ECDC4;">🍶 Glass</span>
                    <span class="class-badge" style="background: #95A5A6;">🥫 Metal</span>
                    <span class="class-badge" style="background: #F39C12;">📄 Paper</span>
                    <span class="class-badge" style="background: #9B59B6;">🍼 Plastic</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Results section (only show if prediction was made)
    if st.session_state.get('prediction_made', False):
        try:
            st.markdown("---")
            st.markdown("## 📊 Detailed Analysis Results")
            
            # Create probability charts
            create_probability_charts(st.session_state['prob_df'])
            
            # Reset session state for next prediction
            if st.button("🔄 Clear Results"):
                for key in ['prediction_made', 'prob_df', 'predictions']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
                
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")
    
    # Dataset overview section with error handling
    try:
        st.markdown("---")
        create_dataset_overview()
    except Exception as e:
        st.error("Error loading dataset overview")
        # Fallback simple stats
        st.markdown("### 📊 Dataset Summary")
        total_images = sum(info['final_count'] for info in CLASS_INFO.values())
        st.write(f"Total Training Images: {total_images:,}")
        st.write("Material Classes: 5 (Cardboard, Glass, Metal, Paper, Plastic)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Training insights
    st.markdown("### 🎓 Training Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="data-insight">
            <h4 style="margin-top: 0; color: #667eea;">🔬 Model Architecture</h4>
            <ul style="margin: 0; padding-left: 1rem;">
                <li><strong>Convolutional Neural Network (CNN)</strong></li>
                <li>3 Conv2D blocks with MaxPooling</li>
                <li>Dropout layers for regularization</li>
                <li>Dense layers for final classification</li>
                <li>Adam optimizer with learning rate 5e-4</li>
                <li>Sparse categorical crossentropy loss</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="data-insight">
            <h4 style="margin-top: 0; color: #667eea;">⚖️ Data Processing</h4>
            <ul style="margin: 0; padding-left: 1rem;">
                <li><strong>Balanced class weights</strong> applied</li>
                <li>80/20 train-test split implemented</li>
                <li>Image normalization: [0, 1] range</li>
                <li>Input standardization: 256×256 RGB</li>
                <li>Data augmentation for robustness</li>
                <li>Early stopping to prevent overfitting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container
    
# Footer with modern styling
    st.markdown("""
    <div class="footer-stats">
        <h3 style="margin-top: 0;">🌱 Making Recycling Smarter with AI</h3>
        <div style="display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin: 2rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700;">2,340+</div>
                <div style="opacity: 0.9;">Training Images</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700;">98%</div>
                <div style="opacity: 0.9;">Classification Accuracy</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700;">5</div>
                <div style="opacity: 0.9;">Material Categories</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700;">♻️</div>
                <div style="opacity: 0.9;">Sustainable Future</div>
            </div>
        </div>
        <p style="margin-bottom: 0; opacity: 0.8;">
            Built with TensorFlow • Streamlit • Love for the Environment 🌍
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    