import streamlit as st
import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import os
from huggingface_hub import hf_hub_download

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glacial Lake Detector",
    page_icon="🏔️",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1c1f26;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #2d3139;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 4px;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load U-Net model — downloads weights from Hugging Face Hub if needed."""
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )

    # Try local first, then Hugging Face Hub
    if os.path.exists('best_unet.pt'):
        weights_path = 'best_unet.pt'
    else:
        with st.spinner('Downloading model weights...'):
            weights_path = hf_hub_download(
                repo_id="YOUR_HF_USERNAME/glacial-lake-detector",  # Update this
                filename="best_unet.pt"
            )

    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

# ── Transforms ────────────────────────────────────────────────────────────────
transform = A.Compose([
    A.Resize(400, 400),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ── Helper functions ───────────────────────────────────────────────────────────
def compute_ndwi(img_array):
    """Compute NDWI from false color image (channels: NIR, Red, Green)"""
    img = img_array.astype(np.float32) / 255.0
    nir   = img[:, :, 0]  # Band 8
    green = img[:, :, 2]  # Band 3
    ndwi  = (green - nir) / (green + nir + 1e-8)
    return (ndwi > 0.0).astype(np.float32)

def predict_mask(model, img_array):
    """Run U-Net inference on image array"""
    img_resized = cv2.resize(img_array, (400, 400))
    augmented   = transform(image=img_resized, mask=np.zeros((400, 400), dtype=np.float32))
    img_tensor  = augmented['image'].unsqueeze(0)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).squeeze().numpy()

    pred_resized = cv2.resize(pred, (img_array.shape[1], img_array.shape[0]))
    return (pred_resized > 0.5).astype(np.float32)

def mask_to_area(mask, image_size_km=10.0):
    """
    Estimate lake area in km².
    Assumes each image covers approximately 10×10 km (typical Sentinel-2 patch).
    """
    total_pixels  = mask.shape[0] * mask.shape[1]
    water_pixels  = mask.sum()
    image_area_km2 = image_size_km ** 2
    return (water_pixels / total_pixels) * image_area_km2

def make_comparison_figure(img_array, ndwi_mask, unet_mask):
    """Generate side-by-side comparison figure"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor('#0e1117')

    titles = ['Input Image', 'NDWI Baseline', 'U-Net Detection', 'Error Map']
    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.axis('off')

    # Input image
    axes[0].imshow(img_array)
    axes[0].set_title(titles[0], color='white', fontsize=13, pad=10)

    # NDWI
    axes[1].imshow(ndwi_mask, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title(titles[1], color='white', fontsize=13, pad=10)

    # U-Net
    axes[2].imshow(unet_mask, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title(titles[2], color='white', fontsize=13, pad=10)

    # Comparison map (U-Net vs NDWI)
    comparison = np.zeros((*unet_mask.shape, 3))
    both       = (unet_mask == 1) & (ndwi_mask == 1)
    unet_only  = (unet_mask == 1) & (ndwi_mask == 0)
    ndwi_only  = (unet_mask == 0) & (ndwi_mask == 1)

    comparison[both]      = [0.20, 0.60, 1.00]  # Blue  — both agree
    comparison[unet_only] = [0.30, 0.85, 0.40]  # Green — U-Net only
    comparison[ndwi_only] = [1.00, 0.40, 0.40]  # Red   — NDWI false positive

    axes[3].imshow(comparison)
    axes[3].set_title(titles[3], color='white', fontsize=13, pad=10)

    legend = [
        mpatches.Patch(color=[0.20, 0.60, 1.00], label='Both agree (water)'),
        mpatches.Patch(color=[0.30, 0.85, 0.40], label='U-Net only'),
        mpatches.Patch(color=[1.00, 0.40, 0.40], label='NDWI false positive'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3,
               fontsize=11, facecolor='#1c1f26', labelcolor='white',
               edgecolor='#2d3139', framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏔️ Himalayan Glacial Lake Detector")
st.markdown("""
**Deep learning pipeline for glacial lake segmentation using Sentinel-2 imagery.**  
Upload a Sentinel-2 false color image (Bands 8, 4, 3) to detect and measure lake area.

*Model: U-Net with ResNet34 encoder · Trained on 410 Himalayan lake images · IoU: 0.9437*
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model Performance**
    | Metric | Score |
    |--------|-------|
    | IoU | **0.9437** |
    | Precision | **0.9761** |
    | Recall | 0.9635 |
    | F1 Score | **0.9693** |

    ---
    **Input Format**
    - Sentinel-2 false color (Bands 8, 4, 3)
    - Any resolution (resized to 400×400)
    - PNG or JPG

    ---
    **What is NDWI?**
    Classical water index:
    `NDWI = (Green - NIR) / (Green + NIR)`
    Shown alongside U-Net for comparison.

    ---
    **GitHub**  
    [View full project →](https://github.com/envie112/Glacial-Lake-Monitoring)
    """)

    st.header("⚙️ Settings")
    image_size_km = st.slider(
        "Image coverage (km)",
        min_value=1.0, max_value=50.0, value=10.0, step=0.5,
        help="Approximate ground coverage of your image in km. Used for area calculation."
    )

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader(
        "Upload Sentinel-2 false color image",
        type=['png', 'jpg', 'jpeg'],
        help="Sentinel-2 false color composite using Bands 8 (NIR), 4 (Red), 3 (Green)"
    )

    st.markdown("---")
    st.subheader("🗺️ Or try a sample")
    use_sample = st.button("Load sample lake image", use_container_width=True)

    if use_sample:
        # Generate a synthetic sample for demo purposes
        sample = np.zeros((400, 400, 3), dtype=np.uint8)
        sample[:, :] = [180, 120, 90]  # Land color
        # Add a fake lake
        cv2.ellipse(sample, (200, 200), (80, 120), 30, 0, 360, (30, 80, 180), -1)
        uploaded = None
        st.session_state['sample_img'] = sample
        st.info("Sample image loaded! Click Detect below.")

with col2:
    if uploaded is not None or 'sample_img' in st.session_state:

        # Load image
        if uploaded is not None:
            img_array = np.array(Image.open(uploaded).convert('RGB'))
            if 'sample_img' in st.session_state:
                del st.session_state['sample_img']
        else:
            img_array = st.session_state['sample_img']

        st.subheader("🖼️ Input Image")
        st.image(img_array, use_column_width=True)

        detect_btn = st.button("🔍 Detect Lake", type="primary", use_container_width=True)

        if detect_btn:
            with st.spinner('Loading model...'):
                model = load_model()

            with st.spinner('Running detection...'):
                ndwi_mask = compute_ndwi(img_array)
                unet_mask = predict_mask(model, img_array)

            # Metrics
            ndwi_area = mask_to_area(ndwi_mask, image_size_km)
            unet_area = mask_to_area(unet_mask, image_size_km)
            ndwi_pct  = ndwi_mask.mean() * 100
            unet_pct  = unet_mask.mean() * 100

            st.subheader("📊 Detection Results")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unet_area:.3f}</div>
                    <div class="metric-label">U-Net Area (km²)</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unet_pct:.1f}%</div>
                    <div class="metric-label">Water Coverage</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{ndwi_area:.3f}</div>
                    <div class="metric-label">NDWI Area (km²)</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                diff = unet_area - ndwi_area
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{diff:+.3f}</div>
                    <div class="metric-label">Difference (km²)</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Comparison figure
            st.subheader("🔬 Visual Comparison")
            fig = make_comparison_figure(img_array, ndwi_mask, unet_mask)
            st.pyplot(fig, use_container_width=True)

            # Interpretation
            st.subheader("💡 Interpretation")
            if unet_pct > 30:
                st.success(f"**Large lake detected** — {unet_pct:.1f}% of image is water ({unet_area:.3f} km²). Significant glacial lake presence.")
            elif unet_pct > 5:
                st.info(f"**Lake detected** — {unet_pct:.1f}% of image is water ({unet_area:.3f} km²). Moderate glacial lake presence.")
            elif unet_pct > 0.5:
                st.warning(f"**Small water body detected** — {unet_pct:.1f}% of image is water ({unet_area:.3f} km²). Small lake or partial detection.")
            else:
                st.error("**No significant water body detected** — less than 0.5% water coverage. Image may not contain a glacial lake.")

            if ndwi_area > unet_area * 1.5:
                st.markdown("> ⚠️ **Note:** NDWI detected significantly more water than U-Net — likely due to snow or ice being misclassified as water. U-Net result is more reliable.")

            # Download mask
            st.subheader("💾 Download")
            mask_img = Image.fromarray((unet_mask * 255).astype(np.uint8))
            buf = io.BytesIO()
            mask_img.save(buf, format='PNG')
            st.download_button(
                label="Download U-Net mask (PNG)",
                data=buf.getvalue(),
                file_name="lake_mask.png",
                mime="image/png",
                use_container_width=True
            )

    else:
        st.info("👆 Upload a Sentinel-2 image or load a sample to get started.")
        st.markdown("""
        **What is Sentinel-2 false color?**

        Sentinel-2 is ESA's free satellite that images the entire Earth every 5 days.
        False color composites use near-infrared (Band 8) instead of blue light — this makes
        water bodies appear distinctly dark while vegetation appears bright red/brown.

        **Where to get Sentinel-2 images:**
        - [Copernicus Browser](https://browser.dataspace.copernicus.eu/) — free, no account needed
        - [Google Earth Engine](https://earthengine.google.com/) — API access
        - [Kaggle Dataset](https://www.kaggle.com/datasets/aatishshresthaa/glacial-lake-dataset) — the dataset used to train this model
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>
    Built with PyTorch · U-Net · Sentinel-2 (ESA Copernicus) · Streamlit<br>
    <a href='https://github.com/envie112/Glacial-Lake-Monitoring' style='color: #4fc3f7;'>GitHub</a> ·
    <a href='https://www.kaggle.com' style='color: #4fc3f7;'>Kaggle Notebook</a>
</div>
""", unsafe_allow_html=True)
