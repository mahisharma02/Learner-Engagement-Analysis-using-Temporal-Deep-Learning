# Learner Engagement Analysis Using Temporal Deep Learning 🎓

## About the Project

Have you ever wondered whether a student attending an online class is actually paying attention or just sitting in front of a screen?

This question inspired us to build **Learner Engagement Analysis Using Temporal Deep Learning**.

Our project aims to analyse a learner's video and estimate their engagement level using deep learning. Instead of looking at just one image, our system studies multiple frames from a video to understand how visual patterns change over time.

We combined computer vision and temporal learning to build a model that predicts engagement into three categories: **Low, High, and Very High**.

## How Does It Work?

1. The user uploads a video of a learner attending an online class.
2. Our system selects 16 frames from the video.
3. Each frame is resized and prepared for the model.
4. MobileNetV2 extracts useful visual information from every frame.
5. The extracted information is arranged in a sequence and passed to a BiLSTM model.
6. BiLSTM learns how the visual information changes across different frames.
7. An Attention mechanism helps the model learn which time steps are more important.
8. Finally, the model predicts whether the learner's engagement is Low, High, or Very High.

## Technologies We Used

- **Python:** Main programming language.
- **TensorFlow and Keras:** Used to build and work with the deep learning model.
- **MobileNetV2:** Used to extract visual features from video frames.
- **BiLSTM:** Used to understand changes and relationships across video frames.
- **Attention Mechanism:** Helps the model assign different importance to different time steps.
- **OpenCV:** Used for reading videos and extracting frames.
- **NumPy and Pandas:** Used for data processing.
- **Scikit-learn:** Used for model evaluation.
- **Streamlit:** Used to create the web application.
- **Google Colab:** Used for model development and experimentation.

## Dataset

We used the **DAiSEE (Dataset for Affective States in E-Environments)** dataset, which contains videos of students participating in online learning activities.

The original dataset provides four engagement levels: Very Low, Low, High, and Very High.

For our experiment, we focused on three categories: Low, High, and Very High. We excluded the Very Low category and used undersampling to address class imbalance in the training data.

Our final balanced training subset contained 639 videos, with 213 videos in each class.

## Model Architecture

Our project combines CNN-based visual feature extraction with LSTM-based temporal learning.

- **MobileNetV2 (CNN):** Extracts visual features from individual video frames.
- **Global Average Pooling:** Converts feature maps into a compact representation of 1,280 values per frame.
- **BiLSTM:** Learns temporal relationships by processing the sequence in both forward and backward directions.
- **Attention:** Learns to assign different importance to different time steps.
- **Softmax:** Produces the final scores for the three engagement categories.

We chose this combination because analysing a video requires understanding both the visual information in individual frames and how that information changes over time.

## Web Application

We developed an interactive Streamlit application to make the project easy to use.

Users can upload a video and get:
- The predicted engagement category.
- Scores for Low, High, and Very High engagement.
- Model evaluation metrics.
- A downloadable JSON report containing prediction details.

### Try the Application

**Live Application:** [Learner Engagement Analysis – Open App](https://exorcist-wok-stitch.ngrok-free.dev)

*Note: The application link is hosted through an ngrok tunnel and may not be available when the Google Colab runtime is disconnected or the tunnel is stopped. The link may also change when a new tunnel is created.*

## What We Learned

Building this project helped us understand how different deep learning concepts can work together in a real-world application.

During development, we explored video preprocessing, CNN-based feature extraction, transfer learning, LSTM, attention mechanisms, class imbalance, model evaluation, and deployment using Streamlit.

We also learned that building a deep learning model is not just about achieving an accuracy score. Data quality, preprocessing, model design, evaluation, and limitations are equally important.

## Future Scope

There is still a lot that can be improved. Some ideas we would like to explore in the future include:

- Training with larger and more diverse datasets.
- Improving prediction performance.
- Experimenting with Transformer-based and other temporal architectures.
- Exploring real-time video analysis.
- Improving the application's interface and usability.

## Important Note

This project provides an experimental estimate of learner engagement based on visible patterns in video recordings. It does not directly measure a person's actual attention, understanding, motivation, or mental state.

---

**Built as an academic deep learning project using MobileNetV2, BiLSTM, and Attention.**