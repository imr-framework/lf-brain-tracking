import tensorflow as tf
import matplotlib.pyplot as plt
from model import DiffusionUNet
from dataset import dataset_generator
from diffusion import add_noise, T
from diffusion import T, get_alpha_schedule, sample_step

# -----------------------
# Parameters
# -----------------------
H, W = 128, 128
DISPLAY_EVERY = 20  # show intermediate steps

# -----------------------
# Load dataset
# -----------------------
ds = tf.data.Dataset.from_generator(
    dataset_generator,
    output_signature=(
        tf.TensorSpec((H, W, 3), tf.float32),
        tf.TensorSpec((H, W, 3), tf.float32)
    )
).batch(1)  # predict one image at a time

# -----------------------
# Load trained model
# -----------------------
model = DiffusionUNet((H, W, 3))
model.load_weights("checkpoints/ddpm.weights.h5")

# -----------------------
# Continuous HF prediction function
# -----------------------
def continuous_predict(model, lf_img, steps=T, display_every=DISPLAY_EVERY):
    """
    Generate HF prediction from LF image using trained DDPM
    """
    x = tf.random.normal(lf_img.shape)  # start from noise
    alpha, alpha_hat = get_alpha_schedule()

    for t in reversed(range(steps)):
        t_tensor = tf.constant([t], dtype=tf.int32)
        eps_theta = model([x, lf_img], training=False)
        x = sample_step(x, eps_theta, t, alpha, alpha_hat)

        if t % display_every == 0 or t == steps-1:
            plt.imshow((x[0, :, :, 1] + 1)/2, cmap='gray')
            plt.title(f"Step {t}")
            plt.axis('off')
            plt.show()

    return x[0].numpy()

# -----------------------
# Predict on first LF image in dataset
# -----------------------
for hf, lf in ds.take(1):
    lf_img = lf.numpy()[0]  # shape (H, W, 3)
    lf_img = lf_img[None, ...]  # add batch dimension
    lf_img = tf.convert_to_tensor(lf_img, dtype=tf.float32)
    hf_pred = continuous_predict(model, lf_img)

# -----------------------
# Visualize final HF prediction
# -----------------------
plt.imshow((hf_pred[:, :, 1] + 1)/2, cmap='gray')
plt.title("Predicted HF")
plt.axis('off')
plt.show()
