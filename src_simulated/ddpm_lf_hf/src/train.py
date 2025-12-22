import tensorflow as tf
from model import DiffusionUNet
from dataset import dataset_generator
from diffusion import add_noise, T
from tensorflow.keras.optimizers import Adam

H, W = 128, 128

model = DiffusionUNet((H, W, 3))
model.compile(optimizer=Adam(1e-4), loss='mse')

ds = tf.data.Dataset.from_generator(
    dataset_generator,
    output_signature=(
        tf.TensorSpec((H, W, 3), tf.float32),
        tf.TensorSpec((H, W, 3), tf.float32)
    )
).shuffle(10).batch(4)

# print("Starting training...")
# print shape of first batch
for hf, lf in ds.take(1):
    print("HF shape:", hf.shape)
    print("LF shape:", lf.shape)

# visualization of first hf and lf slice
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow((hf[0, :, :, 1] + 1) / 2, cmap='gray')
plt.title("HF")
plt.subplot(1, 2, 2)
plt.imshow((lf[0, :, :, 1] + 1) / 2, cmap='gray')
plt.title("LF")
plt.show()

for epoch in range(100):
    for hf, lf in ds:
        t = tf.random.uniform([], 0, T, dtype=tf.int32)
        xt, eps = add_noise(hf, t)
        loss = model.train_on_batch([xt, lf], eps)

    print(f"Epoch {epoch} | Loss {loss:.4f}")
    model.save_weights("checkpoints/ddpm.weights.h5")

print("Training completed.")