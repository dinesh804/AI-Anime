# =========================================================
# AI CARTOON CHARACTER / ANIME AVATAR GAN
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import Adam

# =========================================================
# DATASET PATH
# =========================================================

path = r"E:\Project Y\AI_Anime\data\data"

images = []

# =========================================================
# LOAD IMAGES
# =========================================================

for img in os.listdir(path):

    img_path = os.path.join(path, img)

    # =========================================
    # SKIP FOLDERS
    # =========================================

    if not os.path.isfile(img_path):

        continue

    image = cv2.imread(img_path)

    # =========================================
    # SKIP INVALID IMAGES
    # =========================================

    if image is None:

        continue

    image = cv2.resize(image, (64,64))

    images.append(image)

# =========================================================
# CONVERT TO NUMPY
# =========================================================

images = np.array(images)

# =========================================================
# NORMALIZE IMAGES
# =========================================================

images = images.astype('float32')

images = (images - 127.5) / 127.5

print("\nDATASET SHAPE\n")

print(images.shape)

# =========================================================
# SHOW SAMPLE IMAGE
# =========================================================

plt.figure(figsize=(4,4))

plt.imshow((images[0] + 1) / 2)

plt.title("Sample Anime Face")

plt.axis("off")

plt.show()

# =========================================================
# GENERATOR
# =========================================================

generator = Sequential([

    # =====================================
    # INPUT NOISE VECTOR
    # =====================================

    Dense(

        256 * 8 * 8,

        input_dim=100

    ),

    LeakyReLU(0.2),

    # =====================================
    # RESHAPE
    # =====================================

    Reshape((8,8,256)),

    # =====================================
    # 8x8 -> 16x16
    # =====================================

    Conv2DTranspose(

        128,

        kernel_size=4,

        strides=2,

        padding='same'

    ),

    LeakyReLU(0.2),

    # =====================================
    # 16x16 -> 32x32
    # =====================================

    Conv2DTranspose(

        64,

        kernel_size=4,

        strides=2,

        padding='same'

    ),

    LeakyReLU(0.2),

    # =====================================
    # 32x32 -> 64x64
    # =====================================

    Conv2DTranspose(

        32,

        kernel_size=4,

        strides=2,

        padding='same'

    ),

    LeakyReLU(0.2),

    # =====================================
    # OUTPUT IMAGE
    # =====================================

    Conv2D(

        3,

        kernel_size=3,

        activation='tanh',

        padding='same'

    )

])

print("\nGENERATOR SUMMARY\n")

generator.summary()

# =========================================================
# DISCRIMINATOR
# =========================================================

discriminator = Sequential([

    # =====================================
    # CNN LAYER 1
    # =====================================

    Conv2D(

        64,

        kernel_size=4,

        strides=2,

        padding='same',

        input_shape=(64,64,3)

    ),

    LeakyReLU(0.2),

    Dropout(0.3),

    # =====================================
    # CNN LAYER 2
    # =====================================

    Conv2D(

        128,

        kernel_size=4,

        strides=2,

        padding='same'

    ),

    LeakyReLU(0.2),

    Dropout(0.3),

    # =====================================
    # FLATTEN
    # =====================================

    Flatten(),

    # =====================================
    # OUTPUT
    # =====================================

    Dense(

        1,

        activation='sigmoid'

    )

])

# =========================================================
# COMPILE DISCRIMINATOR
# =========================================================

discriminator.compile(

    optimizer=Adam(0.0002,0.5),

    loss='binary_crossentropy',

    metrics=['accuracy']

)

print("\nDISCRIMINATOR SUMMARY\n")

discriminator.summary()

# =========================================================
# BUILD GAN
# =========================================================

discriminator.trainable = False

gan = Sequential([

    generator,

    discriminator

])

# =========================================================
# COMPILE GAN
# =========================================================

gan.compile(

    optimizer=Adam(0.0002,0.5),

    loss='binary_crossentropy'

)

# =========================================================
# TRAINING PARAMETERS
# =========================================================

epochs = 1000

batch_size = 32

# =========================================================
# TRAIN GAN
# =========================================================

for epoch in range(epochs):

    # =====================================
    # SELECT REAL IMAGES
    # =====================================

    idx = np.random.randint(

        0,

        images.shape[0],

        batch_size

    )

    real_images = images[idx]

    # =====================================
    # GENERATE FAKE IMAGES
    # =====================================

    noise = np.random.normal(

        0,
        1,

        (batch_size,100)

    )

    fake_images = generator.predict(

        noise,

        verbose=0

    )

    # =====================================
    # LABELS
    # =====================================

    real_labels = np.ones(

        (batch_size,1)

    )

    fake_labels = np.zeros(

        (batch_size,1)

    )

    # =====================================
    # TRAIN DISCRIMINATOR
    # =====================================

    d_loss_real = discriminator.train_on_batch(

        real_images,

        real_labels

    )

    d_loss_fake = discriminator.train_on_batch(

        fake_images,

        fake_labels

    )

    d_loss = 0.5 * np.add(

        d_loss_real,

        d_loss_fake

    )

    # =====================================
    # TRAIN GENERATOR
    # =====================================

    noise = np.random.normal(

        0,
        1,

        (batch_size,100)

    )

    g_loss = gan.train_on_batch(

        noise,

        real_labels

    )

    # =====================================
    # PRINT PROGRESS
    # =====================================

    if epoch % 100 == 0:

        print(f"\nEpoch {epoch}")

        print(f"D Loss : {d_loss[0]}")

        print(f"D Accuracy : {d_loss[1] * 100:.2f}%")

        print(f"G Loss : {g_loss}")

# =========================================================
# GENERATE FINAL ANIME FACES
# =========================================================

noise = np.random.normal(

    0,
    1,

    (16,100)

)

generated_images = generator.predict(noise)

# =========================================================
# DISPLAY GENERATED IMAGES
# =========================================================

plt.figure(figsize=(10,10))

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.imshow(

        (generated_images[i] + 1) / 2

    )

    plt.axis("off")
plt.suptitle(
    "AI Generated Cartoon Characters"
)

plt.show()
# =========================================================
# SAVE MODEL
# =========================================================
generator.save("Anime_Avatar_Generator.h5")

print("\nMODEL SAVED SUCCESSFULLY 🚀")