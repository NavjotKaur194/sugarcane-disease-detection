import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import textwrap
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sugarcane AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background: #fbfcf8;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1, h2, h3, h4 {
    color: #294534;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid #d7e2d5;
    background: #ffffff;
    color: #294534;
    font-weight: 600;
    min-height: 42px;
}

.stButton > button:hover {
    border-color: #9eb49f;
    color: #294534;
    background: #f3f8f1;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #bdcdbd;
    border-radius: 18px;
    padding: 10px;
}

img {
    border-radius: 16px;
}

div[data-baseweb="select"] > div {
    border-radius: 12px;
    border-color: #d7e2d5;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HTML HELPER
# ============================================================

def show_html(content):
    st.html(textwrap.dedent(content))


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")


model = load_model()


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow"
]


# ============================================================
# MODEL DATA
# ============================================================

MODEL_DATA = {

    "CNN": {
        "train": 83.77,
        "test": 74.32,
        "description":
            "A conventional Convolutional Neural Network used as "
            "a baseline deep learning architecture for sugarcane "
            "leaf disease classification.",
        "flow":
            "Input → Convolution → Pooling → Feature Extraction → Dense → Classification"
    },

    "ResNet50": {
        "train": 94.50,
        "test": 90.80,
        "description":
            "A deep residual network using skip connections to "
            "learn complex visual features while reducing the "
            "problem of vanishing gradients.",
        "flow":
            "Input → ResNet50 Backbone → Feature Extraction → Global Pooling → Dense → Classification"
    },

    "VGG19": {
        "train": 92.87,
        "test": 87.62,
        "description":
            "A deep convolutional architecture based on stacked "
            "3×3 convolutional layers for hierarchical feature extraction.",
        "flow":
            "Input → VGG19 Convolution Blocks → Feature Extraction → Global Pooling → Dense → Classification"
    },

    "EfficientNetB0": {
        "train": 91.38,
        "test": 86.91,
        "description":
            "An efficient convolutional architecture designed to "
            "achieve a balance between accuracy and computational efficiency.",
        "flow":
            "Input → EfficientNetB0 → Feature Extraction → Global Pooling → Dense → Classification"
    },

    "YOLO + MobileNetV2": {
        "train": 89.47,
        "test": 70.29,
        "description":
            "A lightweight architecture combining YOLO-based detection "
            "with MobileNetV2 for efficient visual analysis.",
        "flow":
            "Input → YOLO Detection → MobileNetV2 Features → Classification"
    }
}


# ============================================================
# DISEASE DATA
# ============================================================

DISEASES = {

    "Healthy": {
        "description":
            "Healthy sugarcane leaf without visible disease symptoms.",
        "details":
            "The leaf generally shows normal green coloration and "
            "does not exhibit characteristic disease lesions."
    },

    "Mosaic": {
        "description":
            "Mosaic disease characterized by irregular light and dark green patterns.",
        "details":
            "Symptoms commonly appear as mottled or mosaic-like "
            "patterns distributed across the leaf surface."
    },

    "RedRot": {
        "description":
            "Red rot disease associated with reddish lesions and tissue damage.",
        "details":
            "The disease can produce reddish or red-brown discoloration "
            "and is an important disease affecting sugarcane."
    },

    "Rust": {
        "description":
            "Rust disease associated with characteristic rust-colored lesions.",
        "details":
            "Small rust-colored pustules or lesions can develop on "
            "the leaf surface."
    },

    "Yellow": {
        "description":
            "Yellowing symptoms associated with affected sugarcane leaves.",
        "details":
            "The leaf develops visible yellow discoloration that can "
            "be associated with disease or stress."
    }
}


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "HOME"

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "img_array" not in st.session_state:
    st.session_state.img_array = None

if "pred_index" not in st.session_state:
    st.session_state.pred_index = None


# ============================================================
# TOP NAVIGATION
# ============================================================

nav_items = [
    "HOME",
    "DETECT",
    "MODELS",
    "PERFORMANCE",
    "EXPLAIN AI",
    "DISEASES",
    "METHODOLOGY",
    "ABOUT"
]

nav_cols = st.columns(
    len(nav_items),
    gap="small"
)

for i, item in enumerate(nav_items):

    with nav_cols[i]:

        if st.button(
            item,
            key=f"nav_{item}",
            use_container_width=True
        ):

            st.session_state.page = item
            st.rerun()


show_html("""
<div style="
    height:1px;
    background:#e3e9e0;
    margin:10px 0 30px 0;
"></div>
""")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    image_rgb = image.convert("RGB")

    resized = image_rgb.resize(
        (224, 224)
    )

    img_array = np.array(
        resized
    ).astype(np.float32)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ResNet50 preprocessing
    img_array = preprocess_input(
        img_array
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        predictions,
        img_array,
        predicted_index
    )


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(
    img_array,
    pred_index
):

    # Saved model contains nested ResNet50
    resnet = model.get_layer(
        "resnet50"
    )

    # Last convolutional feature layer
    last_conv_layer = resnet.get_layer(
        "conv5_block3_out"
    )

    grad_model = tf.keras.models.Model(
        inputs=resnet.input,
        outputs=[
            last_conv_layer.output,
            resnet.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, resnet_output = grad_model(
            img_array
        )

        x = resnet_output

        # Pass through layers after ResNet50
        for layer in model.layers:

            if layer.name != "resnet50":
                x = layer(x)

        predictions = x

        class_channel = predictions[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = (
        conv_outputs
        @ pooled_grads[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return heatmap.numpy()


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):

    original = np.array(
        image.convert("RGB")
    )

    height, width = original.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (width, height)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    return (
        colored_heatmap,
        overlay
    )


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "HOME":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    show_html("""
    <div style="
        background:linear-gradient(
            135deg,
            #e8f2e5 0%,
            #f8f1df 100%
        );
        border:1px solid #dce7d8;
        border-radius:30px;
        padding:55px 60px;
        margin-bottom:42px;
    ">

        <div style="
            color:#66806b;
            font-size:12px;
            font-weight:800;
            letter-spacing:2px;
            margin-bottom:14px;
        ">
            AI-POWERED AGRICULTURAL TECHNOLOGY
        </div>

        <div style="
            color:#294534;
            font-size:48px;
            line-height:1.12;
            font-weight:800;
            margin-bottom:18px;
        ">
            Intelligent Sugarcane Leaf<br>
            Disease Recognition
        </div>

        <div style="
            color:#5f6d63;
            font-size:17px;
            line-height:1.7;
            max-width:760px;
        ">
            A deep learning based research system for automated
            sugarcane leaf disease recognition and comparative
            analysis of multiple deep learning architectures.
        </div>

    </div>
    """)


    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    st.markdown("## 🌱 How It Works")

    st.markdown(
        """
        <p style="
            color:#68766d;
            margin-top:-8px;
            margin-bottom:22px;
        ">
            From image upload to disease prediction in three simple steps.
        </p>
        """,
        unsafe_allow_html=True
    )

    show_html("""
    <div style="
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:20px;
        width:100%;
        margin-bottom:45px;
    ">

        <div style="
            background:#ffffff;
            border:1px solid #e2e7df;
            border-radius:22px;
            padding:28px;
            min-height:205px;
            box-sizing:border-box;
            box-shadow:0 5px 18px rgba(40,60,45,0.05);
        ">

            <div style="
                color:#78927b;
                font-size:12px;
                font-weight:800;
                letter-spacing:1.5px;
            ">
                STEP 01
            </div>

            <div style="
                font-size:30px;
                margin-top:16px;
            ">
                📤
            </div>

            <h3 style="
                color:#294534;
                margin:8px 0;
            ">
                Upload
            </h3>

            <p style="
                color:#68766d;
                font-size:14px;
                line-height:1.6;
                margin:0;
            ">
                Upload a clear image of a sugarcane leaf
                in JPG, JPEG or PNG format.
            </p>

        </div>


        <div style="
            background:#ffffff;
            border:1px solid #e2e7df;
            border-radius:22px;
            padding:28px;
            min-height:205px;
            box-sizing:border-box;
            box-shadow:0 5px 18px rgba(40,60,45,0.05);
        ">

            <div style="
                color:#78927b;
                font-size:12px;
                font-weight:800;
                letter-spacing:1.5px;
            ">
                STEP 02
            </div>

            <div style="
                font-size:30px;
                margin-top:16px;
            ">
                🧠
            </div>

            <h3 style="
                color:#294534;
                margin:8px 0;
            ">
                Analyze
            </h3>

            <p style="
                color:#68766d;
                font-size:14px;
                line-height:1.6;
                margin:0;
            ">
                The image is resized and processed before
                being passed to the trained ResNet50 model.
            </p>

        </div>


        <div style="
            background:#ffffff;
            border:1px solid #e2e7df;
            border-radius:22px;
            padding:28px;
            min-height:205px;
            box-sizing:border-box;
            box-shadow:0 5px 18px rgba(40,60,45,0.05);
        ">

            <div style="
                color:#78927b;
                font-size:12px;
                font-weight:800;
                letter-spacing:1.5px;
            ">
                STEP 03
            </div>

            <div style="
                font-size:30px;
                margin-top:16px;
            ">
                🔬
            </div>

            <h3 style="
                color:#294534;
                margin:8px 0;
            ">
                Predict
            </h3>

            <p style="
                color:#68766d;
                font-size:14px;
                line-height:1.6;
                margin:0;
            ">
                The model predicts the most likely disease
                category along with its confidence score.
            </p>

        </div>

    </div>
    """)


    # --------------------------------------------------------
    # DISEASE CATEGORIES
    # --------------------------------------------------------

    st.markdown("## 🌿 Disease Categories")

    st.markdown(
        """
        <p style="
            color:#68766d;
            margin-top:-8px;
            margin-bottom:22px;
        ">
            Five sugarcane leaf categories recognized by the system.
        </p>
        """,
        unsafe_allow_html=True
    )

    show_html("""
    <div style="
        display:grid;
        grid-template-columns:repeat(5,1fr);
        gap:14px;
        width:100%;
        margin-bottom:15px;
    ">
    """)

    for i, disease in enumerate(CLASS_NAMES):

        show_html(f"""
        <div style="
            background:#fffdf8;
            border:1px solid #ebe6d8;
            border-radius:20px;
            padding:22px 18px;
            height:190px;
            box-sizing:border-box;
        ">

            <div style="
                color:#8a967f;
                font-size:10px;
                font-weight:800;
                letter-spacing:1.2px;
                margin-bottom:12px;
            ">
                CLASS {i + 1}
            </div>

            <div style="
                color:#294534;
                font-size:19px;
                font-weight:800;
                margin-bottom:10px;
            ">
                {disease}
            </div>

            <div style="
                color:#68766d;
                font-size:12.5px;
                line-height:1.55;
            ">
                {DISEASES[disease]["description"]}
            </div>

        </div>
        """)

    show_html("""
    </div>
    """)


    # --------------------------------------------------------
    # RESEARCH MODELS
    # --------------------------------------------------------

    st.markdown("## 🧠 Research Models")

    st.markdown(
        """
        <p style="
            color:#68766d;
            margin-top:-8px;
            margin-bottom:22px;
        ">
            Comparative evaluation of deep learning architectures.
        </p>
        """,
        unsafe_allow_html=True
    )

    show_html("""
    <div style="
        background:linear-gradient(
            135deg,
            #f1f7ee,
            #faf4e7
        );
        border:1px solid #dce7d8;
        border-radius:24px;
        padding:30px 34px;
        margin-bottom:20px;
    ">

        <div style="
            color:#66806b;
            font-size:11px;
            font-weight:800;
            letter-spacing:1.5px;
            margin-bottom:10px;
        ">
            MODEL COMPARISON
        </div>

        <div style="
            color:#294534;
            font-size:27px;
            font-weight:800;
            margin-bottom:10px;
        ">
            Performance Analysis of Deep Learning Architectures
        </div>

        <div style="
            color:#68766d;
            font-size:14px;
            line-height:1.7;
            max-width:900px;
        ">
            The project evaluates CNN, ResNet50, VGG19,
            EfficientNetB0 and YOLO + MobileNetV2 for
            sugarcane leaf disease recognition.
            ResNet50 is used as the deployed model.
        </div>

    </div>
    """)


# ============================================================
# DETECT
# ============================================================

elif st.session_state.page == "DETECT":

    show_html("""
    <div style="margin-bottom:30px;">

        <div style="
            color:#78927b;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
        ">
            AI-POWERED DISEASE CLASSIFICATION
        </div>

        <div style="
            color:#294534;
            font-size:38px;
            font-weight:800;
            margin-top:8px;
        ">
            Sugarcane Leaf Analysis
        </div>

        <div style="
            color:#68766d;
            font-size:15px;
            margin-top:8px;
        ">
            Upload a leaf image and let the trained ResNet50 model
            analyze its disease condition.
        </div>

    </div>
    """)

    left, right = st.columns(
        [1, 1],
        gap="large"
    )


    # --------------------------------------------------------
    # LEFT - UPLOAD
    # --------------------------------------------------------

    with left:

        show_html("""
        <div style="
            background:#ffffff;
            border:1px solid #e0e7dd;
            border-radius:24px;
            padding:28px;
            margin-bottom:20px;
        ">

            <div style="
                color:#294534;
                font-size:20px;
                font-weight:800;
                margin-bottom:8px;
            ">
                Upload Leaf Image
            </div>

            <div style="
                color:#68766d;
                font-size:13px;
            ">
                Supported formats: JPG, JPEG, PNG
            </div>

        </div>
        """)

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            label_visibility="collapsed"
        )


        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.session_state.uploaded_image = image

            st.image(
                image,
                use_container_width=True
            )


            if st.button(
                "Analyze Leaf",
                use_container_width=True
            ):

                (
                    predicted_class,
                    confidence,
                    probabilities,
                    img_array,
                    pred_index
                ) = predict_image(
                    image
                )

                st.session_state.prediction = predicted_class
                st.session_state.confidence = confidence
                st.session_state.probabilities = probabilities
                st.session_state.img_array = img_array
                st.session_state.pred_index = pred_index

                st.rerun()


    # --------------------------------------------------------
    # RIGHT - RESULT
    # --------------------------------------------------------

    with right:

        if (
            st.session_state.prediction is None
            or st.session_state.confidence is None
            or st.session_state.probabilities is None
        ):

            show_html("""
            <div style="
                background:#f5f8f3;
                border:1px solid #dfe8dc;
                border-radius:24px;
                min-height:500px;
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;
                padding:40px;
            ">

                <div>

                    <div style="
                        font-size:45px;
                        margin-bottom:15px;
                    ">
                        🌿
                    </div>

                    <div style="
                        color:#294534;
                        font-size:22px;
                        font-weight:800;
                    ">
                        Ready for Analysis
                    </div>

                    <div style="
                        color:#68766d;
                        font-size:14px;
                        margin-top:10px;
                    ">
                        Upload a sugarcane leaf image
                        and click Analyze Leaf.
                    </div>

                </div>

            </div>
            """)

        else:

            predicted_class = st.session_state.prediction
            confidence = st.session_state.confidence
            probabilities = st.session_state.probabilities


            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            show_html(f"""
            <div style="
                background:#ffffff;
                border:1px solid #dfe7dc;
                border-radius:24px;
                padding:30px;
                margin-bottom:25px;
            ">

                <div style="
                    color:#6e8b72;
                    font-size:11px;
                    font-weight:800;
                    letter-spacing:1.7px;
                ">
                    ✓ ANALYSIS COMPLETE
                </div>

                <div style="
                    color:#294534;
                    font-size:16px;
                    margin-top:22px;
                ">
                    Predicted Condition
                </div>

                <div style="
                    color:#294534;
                    font-size:38px;
                    font-weight:800;
                    margin-top:5px;
                ">
                    {predicted_class}
                </div>

                <div style="
                    color:#78927b;
                    font-size:16px;
                    margin-top:6px;
                ">
                    Confidence: {confidence * 100:.2f}%
                </div>

            </div>
            """)


            # ------------------------------------------------
            # PROBABILITY SECTION
            # ------------------------------------------------

            st.markdown("### Prediction Probabilities")


            for i, disease in enumerate(CLASS_NAMES):

                probability = float(
                    probabilities[i]
                )

                # IMPORTANT:
                # Use st.html through show_html()
                # instead of st.markdown() here.

                show_html(f"""
                <div style="
                    margin-bottom:13px;
                ">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-bottom:5px;
                        font-size:13px;
                    ">

                        <span style="
                            color:#425449;
                            font-weight:600;
                        ">
                            {disease}
                        </span>

                        <span style="
                            color:#718077;
                        ">
                            {probability * 100:.1f}%
                        </span>

                    </div>

                    <div style="
                        width:100%;
                        height:8px;
                        background:#edf1eb;
                        border-radius:20px;
                        overflow:hidden;
                    ">

                        <div style="
                            width:{probability * 100:.2f}%;
                            height:100%;
                            background:#86a68b;
                            border-radius:20px;
                        "></div>

                    </div>

                </div>
                """)


            # ------------------------------------------------
            # EXPLANATION BUTTON
            # ------------------------------------------------

            if st.button(
                "View AI Explanation",
                use_container_width=True
            ):

                st.session_state.page = "EXPLAIN AI"

                st.rerun()


# ============================================================
# MODELS
# ============================================================

elif st.session_state.page == "MODELS":

    show_html("""
    <div style="margin-bottom:30px;">

        <div style="
            color:#78927b;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
        ">
            DEEP LEARNING ARCHITECTURES
        </div>

        <div style="
            color:#294534;
            font-size:38px;
            font-weight:800;
            margin-top:8px;
        ">
            Research Models
        </div>

        <div style="
            color:#68766d;
            font-size:15px;
            margin-top:8px;
        ">
            Comparative analysis of the architectures evaluated
            in the research project.
        </div>

    </div>
    """)


    selected_model = st.selectbox(
        "Select Architecture",
        list(MODEL_DATA.keys())
    )

    data = MODEL_DATA[selected_model]


    left, right = st.columns(
        [1, 1],
        gap="large"
    )


    with left:

        show_html(f"""
        <div style="
            background:#ffffff;
            border:1px solid #e1e8df;
            border-radius:24px;
            padding:30px;
            min-height:350px;
        ">

            <div style="
                color:#6e8b72;
                font-size:11px;
                font-weight:800;
                letter-spacing:1.5px;
            ">
                SELECTED MODEL
            </div>

            <div style="
                color:#294534;
                font-size:32px;
                font-weight:800;
                margin-top:12px;
            ">
                {selected_model}
            </div>

            <div style="
                color:#68766d;
                font-size:15px;
                line-height:1.7;
                margin-top:20px;
            ">
                {data["description"]}
            </div>

        </div>
        """)


    with right:

        show_html(f"""
        <div style="
            background:#f4f8f1;
            border:1px solid #dce7d8;
            border-radius:24px;
            padding:30px;
            min-height:350px;
        ">

            <div style="
                color:#6e8b72;
                font-size:11px;
                font-weight:800;
                letter-spacing:1.5px;
            ">
                ARCHITECTURE FLOW
            </div>

            <div style="
                color:#294534;
                font-size:20px;
                font-weight:700;
                line-height:1.8;
                margin-top:25px;
            ">
                {data["flow"]}
            </div>

            <div style="
                margin-top:35px;
                color:#6e8b72;
                font-size:13px;
                font-weight:700;
            ">
                DEPLOYMENT STATUS
            </div>

            <div style="
                color:#294534;
                font-size:16px;
                margin-top:5px;
            ">
                {"✓ Currently deployed"
                if selected_model == "ResNet50"
                else "Research comparison model"}
            </div>

        </div>
        """)


    st.markdown("### Performance")


    p1, p2 = st.columns(2)


    with p1:

        show_html(f"""
        <div style="
            background:#ffffff;
            border:1px solid #e1e8df;
            border-radius:20px;
            padding:25px;
            text-align:center;
        ">

            <div style="
                color:#78927b;
                font-size:12px;
                font-weight:700;
            ">
                TRAINING ACCURACY
            </div>

            <div style="
                color:#294534;
                font-size:36px;
                font-weight:800;
                margin-top:8px;
            ">
                {data["train"]:.2f}%
            </div>

        </div>
        """)


    with p2:

        show_html(f"""
        <div style="
            background:#ffffff;
            border:1px solid #e1e8df;
            border-radius:20px;
            padding:25px;
            text-align:center;
        ">

            <div style="
                color:#78927b;
                font-size:12px;
                font-weight:700;
            ">
                TESTING ACCURACY
            </div>

            <div style="
                color:#294534;
                font-size:36px;
                font-weight:800;
                margin-top:8px;
            ">
                {data["test"]:.2f}%
            </div>

        </div>
        """)


# ============================================================
# PERFORMANCE
# ============================================================

elif st.session_state.page == "PERFORMANCE":

    show_html("""
    <div style="margin-bottom:30px;">

        <div style="
            color:#78927b;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
        ">
            MODEL EVALUATION
        </div>

        <div style="
            color:#294534;
            font-size:38px;
            font-weight:800;
            margin-top:8px;
        ">
            Performance Analysis
        </div>

        <div style="
            color:#68766d;
            font-size:15px;
            margin-top:8px;
        ">
            Comparison of training and testing accuracy across
            the evaluated deep learning architectures.
        </div>

    </div>
    """)


    model_names = list(MODEL_DATA.keys())

    train_values = [
        MODEL_DATA[m]["train"]
        for m in model_names
    ]

    test_values = [
        MODEL_DATA[m]["test"]
        for m in model_names
    ]


    fig = go.Figure()


    fig.add_trace(
        go.Bar(
            name="Training Accuracy",
            x=model_names,
            y=train_values,
            text=[
                f"{x:.2f}%"
                for x in train_values
            ],
            textposition="outside"
        )
    )


    fig.add_trace(
        go.Bar(
            name="Testing Accuracy",
            x=model_names,
            y=test_values,
            text=[
                f"{x:.2f}%"
                for x in test_values
            ],
            textposition="outside"
        )
    )


    fig.update_layout(
        barmode="group",
        height=520,
        yaxis=dict(
            title="Accuracy (%)",
            range=[0, 105]
        ),
        xaxis=dict(
            title="Model"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            y=1.12
        ),
        margin=dict(
            l=50,
            r=30,
            t=60,
            b=80
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    best_model = max(
        MODEL_DATA,
        key=lambda x: MODEL_DATA[x]["test"]
    )


    show_html(f"""
    <div style="
        background:linear-gradient(
            135deg,
            #eef6eb,
            #faf5e8
        );
        border:1px solid #dce7d8;
        border-radius:24px;
        padding:30px;
        margin-top:15px;
    ">

        <div style="
            color:#78927b;
            font-size:11px;
            font-weight:800;
            letter-spacing:1.5px;
        ">
            BEST PERFORMING MODEL
        </div>

        <div style="
            color:#294534;
            font-size:32px;
            font-weight:800;
            margin-top:8px;
        ">
            {best_model}
        </div>

        <div style="
            color:#68766d;
            font-size:15px;
            margin-top:8px;
        ">
            Testing accuracy:
            {MODEL_DATA[best_model]["test"]:.2f}%
        </div>

    </div>
    """)


# ============================================================
# EXPLAIN AI
# ============================================================

elif st.session_state.page == "EXPLAIN AI":

    show_html("""
    <div style="margin-bottom:30px;">

        <div style="
            color:#78927b;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
        ">
            EXPLAINABLE ARTIFICIAL INTELLIGENCE
        </div>

        <div style="
            color:#294534;
            font-size:38px;
            font-weight:800;
            margin-top:8px;
        ">
            Understanding the Model's Decision
        </div>

        <div style="
            color:#68766d;
            font-size:15px;
            margin-top:8px;
        ">
            Grad-CAM highlights the image regions that contribute
            most strongly to the model's prediction.
        </div>

    </div>
    """)


    if (
        st.session_state.uploaded_image is None
        or st.session_state.prediction is None
        or st.session_state.confidence is None
        or st.session_state.img_array is None
        or st.session_state.pred_index is None
    ):

        show_html("""
        <div style="
            background:#f5f8f3;
            border:1px solid #dfe8dc;
            border-radius:24px;
            padding:50px;
            text-align:center;
        ">

            <div style="
                font-size:45px;
            ">
                🔍
            </div>

            <div style="
                color:#294534;
                font-size:22px;
                font-weight:800;
                margin-top:15px;
            ">
                No Analysis Available
            </div>

            <div style="
                color:#68766d;
                margin-top:8px;
                line-height:1.6;
            ">
                Please go to the DETECT page, upload a sugarcane
                leaf image and click Analyze Leaf first.
            </div>

        </div>
        """)


    else:

        image = st.session_state.uploaded_image

        predicted_class = st.session_state.prediction

        confidence = float(
            st.session_state.confidence
        )

        img_array = st.session_state.img_array

        pred_index = int(
            st.session_state.pred_index
        )


        # ----------------------------------------------------
        # RESULT STRIP
        # ----------------------------------------------------

        show_html(f"""
        <div style="
            background:#f2f7ef;
            border:1px solid #dce7d8;
            border-radius:20px;
            padding:22px 28px;
            margin-bottom:28px;
        ">

            <div style="
                color:#6e8b72;
                font-size:11px;
                font-weight:800;
                letter-spacing:1.5px;
            ">
                MODEL PREDICTION
            </div>

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-top:8px;
            ">

                <div style="
                    color:#294534;
                    font-size:27px;
                    font-weight:800;
                ">
                    {predicted_class}
                </div>

                <div style="
                    color:#66806b;
                    font-size:16px;
                    font-weight:700;
                ">
                    Confidence:
                    {confidence * 100:.2f}%
                </div>

            </div>

        </div>
        """)


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        try:

            heatmap = make_gradcam(
                img_array,
                pred_index
            )

            colored_heatmap, overlay = (
                create_gradcam_overlay(
                    image,
                    heatmap
                )
            )


            c1, c2, c3 = st.columns(
                3,
                gap="medium"
            )


            with c1:

                show_html("""
                <div style="
                    color:#294534;
                    font-size:17px;
                    font-weight:800;
                    margin-bottom:10px;
                ">
                    Original Image
                </div>
                """)

                st.image(
                    image,
                    use_container_width=True
                )


            with c2:

                show_html("""
                <div style="
                    color:#294534;
                    font-size:17px;
                    font-weight:800;
                    margin-bottom:10px;
                ">
                    Grad-CAM Heatmap
                </div>
                """)

                st.image(
                    colored_heatmap,
                    use_container_width=True
                )


            with c3:

                show_html("""
                <div style="
                    color:#294534;
                    font-size:17px;
                    font-weight:800;
                    margin-bottom:10px;
                ">
                    Heatmap Overlay
                </div>
                """)

                st.image(
                    overlay,
                    use_container_width=True
                )


            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            show_html(f"""
            <div style="
                background:#ffffff;
                border:1px solid #e0e7dd;
                border-radius:22px;
                padding:28px;
                margin-top:30px;
            ">

                <div style="
                    color:#78927b;
                    font-size:11px;
                    font-weight:800;
                    letter-spacing:1.5px;
                ">
                    INTERPRETATION
                </div>

                <div style="
                    color:#294534;
                    font-size:18px;
                    font-weight:700;
                    margin-top:10px;
                ">
                    What is the model focusing on?
                </div>

                <p style="
                    color:#68766d;
                    font-size:14px;
                    line-height:1.7;
                    margin-top:10px;
                ">
                    The Grad-CAM visualization highlights the image
                    regions that contributed most strongly to the
                    prediction of <b>{predicted_class}</b>.
                    Brighter regions indicate areas receiving
                    stronger attention from the network.
                </p>

            </div>
            """)


            # ------------------------------------------------
            # HOW TO READ
            # ------------------------------------------------

            st.markdown(
                "### How to Read the Visualization"
            )


            g1, g2, g3 = st.columns(
                3,
                gap="medium"
            )


            with g1:

                show_html("""
                <div style="
                    background:#f8faf6;
                    border:1px solid #e2e8df;
                    border-radius:20px;
                    padding:24px;
                    min-height:150px;
                ">

                    <div style="
                        color:#294534;
                        font-size:17px;
                        font-weight:800;
                    ">
                        Original
                    </div>

                    <p style="
                        color:#68766d;
                        font-size:13px;
                        line-height:1.6;
                    ">
                        The original sugarcane leaf image
                        provided to the model.
                    </p>

                </div>
                """)


            with g2:

                show_html("""
                <div style="
                    background:#f8faf6;
                    border:1px solid #e2e8df;
                    border-radius:20px;
                    padding:24px;
                    min-height:150px;
                ">

                    <div style="
                        color:#294534;
                        font-size:17px;
                        font-weight:800;
                    ">
                        Heatmap
                    </div>

                    <p style="
                        color:#68766d;
                        font-size:13px;
                        line-height:1.6;
                    ">
                        Shows regions receiving stronger
                        activation from the model.
                    </p>

                </div>
                """)


            with g3:

                show_html("""
                <div style="
                    background:#f8faf6;
                    border:1px solid #e2e8df;
                    border-radius:20px;
                    padding:24px;
                    min-height:150px;
                ">

                    <div style="
                        color:#294534;
                        font-size:17px;
                        font-weight:800;
                    ">
                        Overlay
                    </div>

                    <p style="
                        color:#68766d;
                        font-size:13px;
                        line-height:1.6;
                    ">
                        Combines the original image with
                        the Grad-CAM attention regions.
                    </p>

                </div>
                """)


        except Exception as e:

            st.error(
                "Unable to generate Grad-CAM visualization."
            )

            st.exception(e)


# ============================================================
# DISEASES
# ============================================================

elif st.session_state.page == "DISEASES":

    show_html("""
    <div style="margin-bottom:35px;">

        <div style="
            color:#78927b;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
        ">
            SUGARCANE HEALTH
        </div>

        <div style="
            color:#294534;
            font-size:38px;
            font-weight:800;
            margin-top:8px;
        ">
            Disease Categories
        </div>

        <div style="
            color:#68766d;
            font-size:15px;
            margin-top:8px;
        ">
            The system recognizes five sugarcane leaf categories.
        </div>

    </div>
    """)


    for disease in CLASS_NAMES:

        show_html(f"""
        <div style="
            background:#ffffff;
            border:1px solid #e1e8df;
            border-radius:22px;
            padding:28px;
            margin-bottom:18px;
        ">

            <div style="
                color:#294534;
                font-size:24px;
                font-weight:800;
            ">
                {disease}
            </div>

            <div style="
                color:#66806b;
                font-size:14px;
                margin-top:8px;
                font-weight:600;
            ">
                {DISEASES[disease]["description"]}
            </div>

            <div style="
                color:#68766d;
                font-size:14px;
                line-height:1.7;
                margin-top:12px;
            ">
                {DISEASES[disease]["details"]}
            </div>

        </div>
        """)


# ============================================================
# METHODOLOGY
# ============================================================

elif st.session_state.page == "METHODOLOGY":

    show_html("""
    <div style="margin-bottom:35px;">

        <div style="
            color:#78927b;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
        ">
            RESEARCH PIPELINE
        </div>

        <div style="
            color:#294534;
            font-size:38px;
            font-weight:800;
            margin-top:8px;
        ">
            Methodology
        </div>

        <div style="
            color:#68766d;
            font-size:15px;
            margin-top:8px;
        ">
            End-to-end workflow followed in the research project.
        </div>

    </div>
    """)


    steps = [

        (
            "01",
            "Dataset",
            "Sugarcane leaf images are organized into five disease categories."
        ),

        (
            "02",
            "Preprocessing",
            "Images are resized to 224 × 224 pixels and processed according to the selected architecture."
        ),

        (
            "03",
            "Model Training",
            "Multiple deep learning architectures are trained and evaluated for disease recognition."
        ),

        (
            "04",
            "Performance Evaluation",
            "Training and testing performance are compared using accuracy and classification results."
        ),

        (
            "05",
            "Model Selection",
            "ResNet50 provides the strongest testing performance among the compared models and is selected for deployment."
        ),

        (
            "06",
            "Application",
            "The trained model is integrated into a Streamlit application for image-based prediction."
        ),

        (
            "07",
            "Explainability",
            "Grad-CAM is used to visualize image regions influencing the model prediction."
        )

    ]


    for number, title, description in steps:

        show_html(f"""
        <div style="
            display:flex;
            gap:22px;
            align-items:flex-start;
            background:#ffffff;
            border:1px solid #e1e8df;
            border-radius:20px;
            padding:24px;
            margin-bottom:14px;
        ">

            <div style="
                min-width:48px;
                height:48px;
                border-radius:14px;
                background:#edf5e9;
                color:#66806b;
                display:flex;
                align-items:center;
                justify-content:center;
                font-weight:800;
            ">
                {number}
            </div>

            <div>

                <div style="
                    color:#294534;
                    font-size:19px;
                    font-weight:800;
                ">
                    {title}
                </div>

                <div style="
                    color:#68766d;
                    font-size:14px;
                    line-height:1.7;
                    margin-top:5px;
                ">
                    {description}
                </div>

            </div>

        </div>
        """)


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "ABOUT":

    show_html("""
    <div style="margin-bottom:35px;">

        <div style="
            color:#78927b;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
        ">
            ABOUT THE PROJECT
        </div>

        <div style="
            color:#294534;
            font-size:38px;
            font-weight:800;
            margin-top:8px;
        ">
            Sugarcane AI
        </div>

    </div>
    """)


    show_html("""
    <div style="
        background:#ffffff;
        border:1px solid #e1e8df;
        border-radius:24px;
        padding:32px;
        margin-bottom:25px;
    ">

        <div style="
            color:#294534;
            font-size:23px;
            font-weight:800;
        ">
            Performance Analysis of Deep Learning Architectures
            for Sugarcane Leaf Disease Recognition
        </div>

        <p style="
            color:#68766d;
            font-size:15px;
            line-height:1.8;
            margin-top:16px;
        ">
            This project investigates the application of deep learning
            for automated sugarcane leaf disease recognition.
            Multiple architectures are evaluated to understand their
            classification performance and identify a suitable model
            for deployment.
        </p>

        <p style="
            color:#68766d;
            font-size:15px;
            line-height:1.8;
        ">
            The evaluated architectures include CNN, ResNet50, VGG19,
            EfficientNetB0 and YOLO + MobileNetV2. The deployed
            application uses ResNet50 for image-based disease prediction.
        </p>

    </div>
    """)


    a1, a2 = st.columns(
        2,
        gap="large"
    )


    with a1:

        show_html("""
        <div style="
            background:#f4f8f1;
            border:1px solid #dce7d8;
            border-radius:22px;
            padding:28px;
            min-height:250px;
        ">

            <div style="
                color:#294534;
                font-size:20px;
                font-weight:800;
            ">
                Technologies
            </div>

            <p style="
                color:#68766d;
                font-size:14px;
                line-height:1.9;
            ">
                • Python<br>
                • TensorFlow / Keras<br>
                • ResNet50<br>
                • OpenCV<br>
                • NumPy<br>
                • Streamlit<br>
                • Plotly
            </p>

        </div>
        """)


    with a2:

        show_html("""
        <div style="
            background:#fffaf0;
            border:1px solid #ebe2cf;
            border-radius:22px;
            padding:28px;
            min-height:250px;
        ">

            <div style="
                color:#294534;
                font-size:20px;
                font-weight:800;
            ">
                Explainable AI
            </div>

            <p style="
                color:#68766d;
                font-size:14px;
                line-height:1.8;
            ">
                Grad-CAM is integrated into the application to provide
                visual explanations of model predictions. The technique
                highlights image regions that contribute strongly to
                the classification decision.
            </p>

        </div>
        """)


    st.markdown(
        "### Project Objective"
    )


    show_html("""
    <div style="
        background:linear-gradient(
            135deg,
            #eef6eb,
            #faf5e8
        );
        border:1px solid #dce7d8;
        border-radius:22px;
        padding:30px;
    ">

        <p style="
            color:#526157;
            font-size:15px;
            line-height:1.8;
            margin:0;
        ">
            The primary objective is to analyze and compare deep learning
            architectures for sugarcane leaf disease recognition and
            develop an interactive application capable of providing
            predictions together with visual model explanations.
        </p>

    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

show_html("""
<div style="
    border-top:1px solid #e3e9e0;
    margin-top:55px;
    padding-top:20px;
    text-align:center;
    color:#89958c;
    font-size:12px;
">
    Sugarcane AI · Deep Learning Based Disease Recognition ·
    Explainable Artificial Intelligence
</div>
""")