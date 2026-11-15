
# ============================================================
# USER INTERFACE
# ============================================================

import subprocess

CLASS_NAMES = ["Low", "High", "Very High"]

# Previously calculated on the DAiSEE test set
MODEL_METRICS = {
    "Accuracy": 38.37,
    "Balanced Accuracy": 32.85,
    "Macro F1 Score": 23.89,
    "Weighted F1 Score": 29.54
}

st.set_page_config(
    page_title="Learner Engagement Analyzer",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
    .main {
        background-color: #f6f7f9;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #172b4d;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-card {
        background: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 10px;
    }

    div.stButton > button {
        background-color: #244b74;
        color: white;
        border-radius: 8px;
        border: none;
        height: 48px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #183653;
        color: white;
    }

    .footer {
        color: #64748b;
        text-align: center;
        font-size: 13px;
        padding-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.title("🎓 Learner Engagement Analyzer")

st.markdown(
    '<p class="subtitle">'
    'Video-based analysis of observable learner engagement '
    'indicators using a trained DAiSEE deep learning model.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- MODEL LOADING ----------------

try:
    with st.spinner("Preparing the trained model..."):
        model, extractor = load_models()

    st.success("Model is ready for video analysis.")

except Exception as e:
    st.error("Could not load the trained model.")
    st.exception(e)
    st.stop()


# ---------------- MODEL PERFORMANCE ----------------

st.subheader("Model Performance")

st.caption(
    "Evaluation results on the held-out DAiSEE test set. "
    "These are model-level metrics, not scores for an individual video."
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Accuracy", f'{MODEL_METRICS["Accuracy"]:.2f}%')

with m2:
    st.metric(
        "Balanced Accuracy",
        f'{MODEL_METRICS["Balanced Accuracy"]:.2f}%'
    )

with m3:
    st.metric(
        "Macro F1 Score",
        f'{MODEL_METRICS["Macro F1 Score"]:.2f}%'
    )

with m4:
    st.metric(
        "Weighted F1 Score",
        f'{MODEL_METRICS["Weighted F1 Score"]:.2f}%'
    )

with st.expander("About the evaluation metrics"):
    st.write("""
    - **Accuracy:** Overall proportion of correct predictions.
    - **Balanced Accuracy:** Average recall across the three classes.
    - **Macro F1:** Unweighted average of the F1-scores of all classes.
    - **Weighted F1:** Average F1-score weighted by class support.
    """)

st.divider()


# ---------------- VIDEO PREVIEW FUNCTION ----------------

def create_browser_preview(input_path):

    preview_path = input_path.rsplit(".", 1)[0] + "_preview.mp4"

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                preview_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )

        if result.returncode == 0 and os.path.exists(preview_path):
            return preview_path

    except Exception:
        pass

    return None


# ---------------- VIDEO UPLOAD ----------------

st.subheader("Video Analysis")

st.write(
    "Select a student video to analyze its engagement-related "
    "behavioral indicators."
)

uploaded_file = st.file_uploader(
    "Upload video",
    type=["avi", "mp4", "mov", "mkv"],
    help="Supported formats: AVI, MP4, MOV and MKV."
)

if uploaded_file is not None:

    st.info(f"Selected file: {uploaded_file.name}")

    original_temp_path = None
    preview_path = None

    try:
        # Save uploaded video temporarily
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            original_temp_path = temp_file.name

        st.write("**Video preview**")

        # Convert AVI/MOV/MKV to browser-compatible MP4
        if suffix.lower() != ".mp4":

            with st.spinner("Preparing video preview..."):
                preview_path = create_browser_preview(
                    original_temp_path
                )

            if preview_path:
                st.video(preview_path)
            else:
                st.warning(
                    "Browser preview could not be generated. "
                    "The original video can still be analyzed."
                )

        else:
            st.video(uploaded_file.getvalue())

        st.write("")

        analyze = st.button(
            "Analyze Video",
            type="primary",
            use_container_width=True
        )

        if analyze:

            with st.spinner(
                "Extracting frames and analyzing the video..."
            ):

                # Extract 16-frame feature sequence
                features = extract_video_features(
                    original_temp_path,
                    extractor
                )

                if features.shape != (16, 1280):
                    raise ValueError(
                        f"Unexpected feature shape: {features.shape}"
                    )

                sequence = np.expand_dims(
                    features.astype(np.float32),
                    axis=0
                )

                # Model prediction
                probabilities = model.predict(
                    sequence,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = CLASS_NAMES[predicted_index]

                top_probability = float(
                    probabilities[predicted_index]
                ) * 100

            # ---------------- RESULTS ----------------

            st.divider()
            st.subheader("Analysis Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Predicted Engagement Category",
                    predicted_class
                )

            with result_col2:
                st.metric(
                    "Top Class Probability",
                    f"{top_probability:.2f}%"
                )

            st.subheader("Class-wise Probability")

            chart_data = {
                "Category": CLASS_NAMES,
                "Probability (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilities
                ]
            }

            st.bar_chart(
                chart_data,
                x="Category",
                y="Probability (%)",
                color="#376b91"
            )

            for name, probability in zip(
                CLASS_NAMES,
                probabilities
            ):

                percentage = float(probability) * 100

                st.write(f"**{name} — {percentage:.2f}%**")

                st.progress(
                    min(max(float(probability), 0.0), 1.0)
                )

            # ---------------- REPORT DOWNLOAD ----------------

            report = {
                "video": uploaded_file.name,
                "predicted_category": predicted_class,
                "top_class_probability": round(
                    top_probability, 2
                ),
                "class_probabilities": {
                    name: round(float(p) * 100, 2)
                    for name, p in zip(
                        CLASS_NAMES,
                        probabilities
                    )
                },
                "model": "MobileNetV2_BiLSTM_Attention",
                "dataset": "DAiSEE",
                "model_test_metrics": MODEL_METRICS
            }

            st.download_button(
                "Download Analysis Report",
                data=json.dumps(report, indent=4),
                file_name="learner_engagement_report.json",
                mime="application/json",
                use_container_width=True
            )

            st.caption(
                "The prediction represents observable behavioral "
                "indicators learned from DAiSEE annotations. "
                "It does not directly measure a student's internal "
                "attention, feelings or mental state."
            )

    except Exception as e:
        st.error("Video analysis could not be completed.")
        st.exception(e)

    finally:

        if original_temp_path and os.path.exists(original_temp_path):
            os.remove(original_temp_path)

        if preview_path and os.path.exists(preview_path):
            os.remove(preview_path)


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    '<div class="footer">'
    'Learner Engagement Analyzer | DAiSEE Dataset | '
    'Three-Class Deep Learning Model'
    '</div>',
    unsafe_allow_html=True
)
