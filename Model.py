# importing modules

from typing import List
from keras.layers import (
    Activation,
    Dense,
    BatchNormalization,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    Input,
    Reshape,
)
from keras.models import Model, Sequential
from keras.optimizers import Adam


class Fcn_Network:
    """
    class defining our Fcn segmentation network
    """

    def __init__(self, epochs, model_save_name, lr) -> None:
        self.EPOCHS = epochs
        self.SAVE_NAME = model_save_name
        self.LR = lr

    def model_init(
        self, x_train, y_train, x_val, y_val, loss: List, metrics: List
    ) -> tuple(keras.models.Model, keras.models.Model):  # type: ignore
        # Convolution Layers (BatchNorm after non-linear activation)

        img_input = Input(shape=(192, 256, 3))
        x = Conv2D(16, (5, 5), padding="same", name="conv1", strides=(1, 1))(img_input)
        x = BatchNormalization(name="bn1")(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same", name="conv2")(x)
        x = BatchNormalization(name="bn2")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D()(x)
        x = Conv2D(64, (4, 4), padding="same", name="conv3")(x)
        x = BatchNormalization(name="bn3")(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (4, 4), padding="same", name="conv4")(x)
        x = BatchNormalization(name="bn4")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D()(x)

        x = Dropout(0.5)(x)

        x = Conv2D(512, (3, 3), padding="same", name="conv5")(x)
        x = BatchNormalization(name="bn5")(x)
        x = Activation("relu")(x)
        x = Dense(1024, activation="relu", name="fc1")(x)
        x = Dense(1024, activation="relu", name="fc2")(x)

        # Deconvolution Layers (BatchNorm after non-linear activation)

        x = Conv2DTranspose(256, (3, 3), padding="same", name="deconv1")(x)
        x = BatchNormalization(name="bn6")(x)
        x = Activation("relu")(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(256, (3, 3), padding="same", name="deconv2")(x)
        x = BatchNormalization(name="bn7")(x)
        x = Activation("relu")(x)
        x = Conv2DTranspose(128, (3, 3), padding="same", name="deconv3")(x)
        x = BatchNormalization(name="bn8")(x)
        x = Activation("relu")(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(1, (3, 3), padding="same", name="deconv4")(x)
        x = BatchNormalization(name="bn9")(x)

        x = Dropout(0.5)(x)

        x = Activation("sigmoid")(x)
        pred = Reshape((192, 256))(x)

        model = Model(inputs=img_input, outputs=pred)

        model.compile(optimizer=Adam(lr=self.LR), loss=loss, metrics=metrics)

        hist = model.fit(
            x_train,
            y_train,
            epochs=self.EPOCHS,
            batch_size=18,
            validation_data=(x_val, y_val),
            verbose=1,
        )

        model.save(self.SAVE_NAME)
        return model, hist
