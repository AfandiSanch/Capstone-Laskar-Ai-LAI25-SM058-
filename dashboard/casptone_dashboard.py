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
    page_title="🗂️ Garbage Classification Dashboard",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .class-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

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
            <div class="warning-box">
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
            <div class="success-box">
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

# Class information based on your notebook data (exactly matching your notebook counts)
CLASS_INFO = {
    'cardboard': {
        'original_count': 393, 
        'final_count': 314,  # After train/test split (training set)
        'color': '#8B4513', 
        'description': 'Recyclable cardboard materials including boxes and packaging'
    },
    'glass': {
        'original_count': 491, 
        'final_count': 392,  # After train/test split (training set)
        'color': '#00CED1', 
        'description': 'Glass bottles, jars, and containers'
    },
    'metal': {
        'original_count': 400, 
        'final_count': 320,  # After train/test split (training set)
        'color': '#C0C0C0', 
        'description': 'Metal cans, aluminum containers, and metallic objects'
    },
    'paper': {
        'original_count': 584, 
        'final_count': 467,  # After train/test split (training set)
        'color': '#F5DEB3', 
        'description': 'Paper documents, newspapers, and paper materials'
    },
    'plastic': {
        'original_count': 472, 
        'final_count': 377,  # After train/test split (training set)
        'color': '#FF6347', 
        'description': 'Plastic bottles, containers, and plastic waste'
    }
}

# Class names exactly as in notebook
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# Function to inspect model architecture
def inspect_model(model):
    """Display model architecture information for debugging."""
    if model is not None:
        st.markdown("### 🔍 Model Architecture Debug Info")
        
        # Expected architecture from notebook:
        st.markdown("""
        **Expected Model Architecture (from notebook):**
        ```
        Conv2D(32, (3,3)) + MaxPooling2D
        Conv2D(64, (3,3)) + MaxPooling2D + Dropout(0.25)
        Conv2D(128, (3,3)) + MaxPooling2D + Dropout(0.25)
        Flatten
        Dense(256) + Dropout(0.4)
        Dense(5, softmax)
        ```
        **Input Shape:** (256, 256, 3)
        **Optimizer:** Adam(lr=5e-4)
        **Loss:** sparse_categorical_crossentropy
        """)
        
        # Create a summary
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        summary_text = '\n'.join(summary_list)
        
        st.code(summary_text, language='text')
        
        # Show layer details
        st.markdown("#### Layer Details:")
        for i, layer in enumerate(model.layers):
            st.write(f"**Layer {i+1}**: {layer.name} ({type(layer).__name__})")
            if hasattr(layer, 'output_shape'):
                st.write(f"  - Output shape: {layer.output_shape}")
            if hasattr(layer, 'input_shape'):
                st.write(f"  - Input shape: {layer.input_shape}")

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🗂️ Recyclable Materials Classification Dashboard</div>', unsafe_allow_html=True)
    
    # Check model status
    model_status = "sequential.h5" if os.path.exists("sequential.h5") else "not found"
    
    if model_status == "sequential.h5":
        st.markdown("""
        <div class="success-box">
            ✅ <strong>Model Status:</strong> Your trained model (sequential.h5) is ready to use!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Model Status:</strong> Will attempt to download from Google Drive on first prediction.
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dataset Overview")
        st.markdown('<div class="class-info">', unsafe_allow_html=True)
        st.markdown("**Recyclable Materials Classification Dataset**")
        st.markdown("Dataset processed exactly as in your notebook:")
        
        # Show original counts and final training counts
        total_original = sum(info['original_count'] for info in CLASS_INFO.values())
        total_final = sum(info['final_count'] for info in CLASS_INFO.values())
        
        for class_name, info in CLASS_INFO.items():
            original_pct = (info['original_count'] / total_original) * 100
            final_pct = (info['final_count'] / total_final) * 100
            
            st.markdown(f"• **{class_name.title()}**: {info['original_count']} → {info['final_count']} training images ({final_pct:.1f}%)")
        
        st.markdown(f"**Original Total**: {total_original}")
        st.markdown(f"**Training Set**: {total_final} (after 80/20 split)")
        st.markdown(f"**Trash class**: Removed as per notebook")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Model Architecture")
        st.info("""
        **CNN Model Details (from notebook):**
        - Input: 256×256×3 RGB images
        - 3 Conv2D blocks (32→64→128 filters)
        - MaxPooling + Dropout layers
        - Dense(256) + Dropout(0.4)
        - Output: Dense(5, softmax)
        - Optimizer: Adam(lr=5e-4)
        - Loss: sparse_categorical_crossentropy
        """)
        
        st.markdown("### ⚖️ Class Weights")
        st.info("""
        **Balanced class weights applied:**
        - Computed using sklearn's 'balanced' method
        - Glass & Metal boosted by 1.5x priority
        - Handles class imbalance automatically
        """)
        
        st.markdown("---")
        st.markdown("### 📥 Model Management")
        if st.button("🔄 Re-download Model"):
            if os.path.exists("sequential.h5"):
                os.remove("sequential.h5")
            st.cache_resource.clear()
            st.rerun()
            
        # Debug mode
        if st.checkbox("🔍 Debug Mode (Show Model Architecture)"):
            st.session_state.debug_mode = True
        else:
            st.session_state.debug_mode = False
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="subheader">📤 Upload Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of recyclable materials to classify (will be resized to 256×256 as per notebook)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Show image info
            st.markdown(f"**Image Info:** {image.size[0]}×{image.size[1]} pixels, Mode: {image.mode}")
            st.info("Image will be resized to 256×256 and normalized to [0,1] range (matching notebook preprocessing)")
            
            # Load model
            with st.spinner("Loading model..."):
                model = load_model()
            
            # Debug mode - show model architecture
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                if model is not None:
                    inspect_model(model)
            
            if model is not None:
                # Make prediction
                with st.spinner("Analyzing image..."):
                    processed_image = preprocess_image(image)
                    predictions = predict_image(model, processed_image)
                
                if predictions is not None:
                    # Ensure we have the right number of predictions
                    if len(predictions) >= len(CLASS_NAMES):
                        predictions = predictions[:len(CLASS_NAMES)]  # Take only first 5
                    else:
                        st.error(f"Model returned {len(predictions)} predictions, but we need {len(CLASS_NAMES)}")
                        st.stop()
                    
                    # Get top prediction
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    confidence = predictions[predicted_class_idx] * 100
                    
                    # Display main prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>♻️ Classification Result</h2>
                        <h1>{predicted_class.upper()}</h1>
                        <h3>Confidence: {confidence:.2f}%</h3>
                        <p>Recyclable Material Classification</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Model could not be loaded. Please check the model file.")
    
    with col2:
        if uploaded_file is not None and 'predictions' in locals() and predictions is not None:
            st.markdown('<div class="subheader">📈 Prediction Probabilities</div>', unsafe_allow_html=True)
            
            # Create probability dataframe
            prob_df = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Probability': predictions * 100,
                'Color': [CLASS_INFO[cls]['color'] for cls in CLASS_NAMES]
            }).sort_values('Probability', ascending=True)
            
            # Bar chart
            fig_bar = px.bar(
                prob_df, 
                x='Probability', 
                y='Class',
                color='Color',
                color_discrete_map={color: color for color in prob_df['Color']},
                title="Class Probabilities (%)",
                labels={'Probability': 'Probability (%)', 'Class': 'Material Type'},
                orientation='h'
            )
            fig_bar.update_layout(
                showlegend=False,
                height=350,
                xaxis_range=[0, 100]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Pie chart
            fig_pie = px.pie(
                prob_df,
                values='Probability',
                names='Class',
                title="Probability Distribution",
                color='Class',
                color_discrete_map={cls: CLASS_INFO[cls]['color'] for cls in CLASS_NAMES}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Detailed metrics
            st.markdown('<div class="subheader">📋 Detailed Results</div>', unsafe_allow_html=True)
            
            sorted_indices = np.argsort(predictions)[::-1]  # Sort by probability descending
            
            for rank, idx in enumerate(sorted_indices):
                class_name = CLASS_NAMES[idx]
                prob = predictions[idx]
                
                with st.expander(f"#{rank+1} {class_name.title()} - {prob*100:.2f}%"):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.metric("Probability", f"{prob*100:.4f}%")
                        st.metric("Rank", f"#{rank+1}")
                        st.metric("Logit Score", f"{prob:.6f}")
                    with col_b:
                        st.write(f"**Description**: {CLASS_INFO[class_name]['description']}")
                        st.write(f"**Original samples**: {CLASS_INFO[class_name]['original_count']}")
                        st.write(f"**Training samples**: {CLASS_INFO[class_name]['final_count']}")
                        
                        # Progress bar
                        st.progress(float(prob))
        else:
            st.markdown('<div class="subheader">👆 Upload an image to see classification</div>', unsafe_allow_html=True)
            st.info("Please upload an image file (PNG, JPG, or JPEG) to get started with recyclable materials classification.")
            
            # Dataset visualization
            st.markdown("### 📊 Training Dataset Distribution")
            
            # Create dataset overview chart
            dataset_df = pd.DataFrame([
                {
                    'Class': cls, 
                    'Original Count': info['original_count'], 
                    'Training Set': info['final_count'],
                    'Color': info['color']
                } 
                for cls, info in CLASS_INFO.items()
            ])
            
            # Melt the dataframe for grouped bar chart
            dataset_melted = pd.melt(dataset_df, 
                                   id_vars=['Class', 'Color'], 
                                   value_vars=['Original Count', 'Training Set'],
                                   var_name='Dataset', value_name='Count')
            
            fig_dataset = px.bar(
                dataset_melted,
                x='Class',
                y='Count',
                color='Dataset',
                title="Training Dataset Distribution (Original vs Final Training Set)",
                barmode='group'
            )
            fig_dataset.update_layout(showlegend=True)
            st.plotly_chart(fig_dataset, use_container_width=True)
            
            # Show processing details
            st.markdown("### 🔧 Data Processing (Matching Notebook)")
            st.success("""
            **Processing steps applied:**
            1. ✅ Removed 'trash' class (as per notebook)
            2. ✅ Applied 80/20 train/test split
            3. ✅ Used balanced class weights
            4. ✅ Priority boost for glass & metal (1.5x)
            5. ✅ Image normalization: rescale=1.0/255.0
            6. ✅ Target size: 256×256 pixels
            7. ✅ Validation split: 20% of training data
            """)
            
            st.markdown("### 🎯 Model Training Details")
            st.info("""
            **Training configuration:**
            - **Epochs**: Up to 30 (with early stopping)
            - **Batch size**: 32
            - **Optimizer**: Adam (learning_rate=5e-4)
            - **Loss function**: sparse_categorical_crossentropy
            - **Early stopping**: patience=5, monitor=val_accuracy
            - **Class weights**: Balanced (computed automatically)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>♻️ Recyclable Materials Classification Dashboard | Built with Streamlit & TensorFlow</p>
        <p>Model Architecture: CNN with 256×256 input | 5 Recyclable Material Classes | Balanced Class Weights 🌱</p>
        <p><strong>Note:</strong> Dashboard aligned exactly with notebook preprocessing and architecture</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()