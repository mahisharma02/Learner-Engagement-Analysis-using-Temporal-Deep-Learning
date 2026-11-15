import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def sample_video(video_path, num_frames=16, image_size=(224,224)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames")
    indices = np.linspace(0, count-1, num_frames).astype(int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Could not read frame {i}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size)
        frames.append(frame.astype("float32"))
    cap.release()
    return np.asarray(frames)

def prepare_frames(frames):
    return preprocess_input(frames.copy())
