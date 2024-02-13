<body>
    <h1>Pokemon GAN Project</h1>
    <p>This project explores the creation of a Generative Adversarial Network (GAN) designed to generate new Pokemon images. Utilizing TensorFlow and Keras, the project builds and trains a deep learning model capable of producing images that mimic the style of Pokemon characters from a dataset of existing Pokemon images.</p>
    <h2>Project Setup and Data Preparation</h2>
    <p>The project begins with the setup of necessary Python libraries and the extraction of Pokemon images from a provided dataset. Images are preprocessed to normalize their pixel values, preparing them for input into the GAN model.</p>
    <pre>
    <code>
    # Import necessary libraries
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    from tqdm.notebook import tqdm
    # Load and preprocess the dataset
    image_paths = [os.path.join('/content/tmp', name) for name in os.listdir('/content/tmp')]
    train_images = np.array([np.array(Image.open(path)) for path in tqdm(image_paths)])
    train_images = (train_images - 127.5) / 127.5
    </code>
    </pre>
    <h2>Model Architecture</h2>
    <h3>Generator Model</h3>
    <pre>
    <code>
    # Generator architecture
    generator = keras.Sequential([
        keras.layers.Dense(8*8*512, input_dim=100, activation='relu'),
        keras.layers.Reshape((8, 8, 512)),
        keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'),
        keras.layers.Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')
    ], name='generator')
    </code>
    </pre>
    <h3>Discriminator Model</h3>
    <pre>
    <code>
    # Discriminator architecture
    discriminator = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(64, 64, 3), activation=keras.layers.LeakyReLU(alpha=0.2)),
        keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation=keras.layers.LeakyReLU(alpha=0.2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ], name='discriminator')
    </code>
    </pre>
    <h2>Training Process</h2>
    <p>The DCGAN model is trained by alternately training the discriminator and generator models. The discriminator is trained to correctly classify real and fake images, while the generator is trained to produce images that the discriminator classifies as real.</p>
    <pre>
    <code>
    # Compile and train the DCGAN model
    dcgan = DCGAN(generator=generator, discriminator=discriminator, latent_dim=100)
    dcgan.compile(g_optimizer=keras.optimizers.Adam(0.0002, 0.5),
                  d_optimizer=keras.optimizers.Adam(0.0002, 0.5),
                  loss_fn=keras.losses.BinaryCrossentropy())
    dcgan.fit(train_images, epochs=50, callbacks=[DCGANMonitor(latent_dim=100)])
    </code>
    </pre>
    <h2>Image Generation</h2>
    <p>After training, the generator model is capable of producing new Pokemon images from random noise. These images are visualized to evaluate the performance and creativity of the GAN.</p>
    <pre>
    <code>
    # Generate a new Pokemon image
    noise = tf.random.normal([1, 100])
    generated_image = dcgan.generator(noise, training=False)
    plt.imshow((generated_image[0] * 127.5 + 127.5).numpy().astype('uint8'))
    plt.axis('off')
    plt.show()
    </code>
    </pre>
    <h2>Conclusion</h2>
    <p>This project demonstrates the power of GANs in generating new content, specifically new Pokemon images, showcasing the potential for deep learning models in creative applications. Through careful model design and training, the project achieves its goal of generating visually compelling images that resemble the iconic Pokemon characters.</p>
</body>
