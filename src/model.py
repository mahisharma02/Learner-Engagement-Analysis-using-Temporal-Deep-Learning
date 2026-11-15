import tensorflow as tf
from tensorflow.keras import layers, Model

def build_encoder():
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False
    inp = layers.Input((224,224,3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    return Model(inp, x, name="mobilenetv2_encoder")

def build_temporal_model(num_frames=16, feature_dim=1280, num_classes=4):
    inp = layers.Input((num_frames, feature_dim))
    x = layers.LSTM(128, dropout=0.3)(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out, name="cnn_lstm_engagement")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
