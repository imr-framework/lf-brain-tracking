import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your data (Change 'losses.xlsx' to your actual file name)
df = pd.read_csv('src_simulated/outputs/cyclegan_t1_t2_upsample5/training_metrics.csv')

# 2. Smoothing function (Window size 50 helps see the trend through the GAN noise)
def smooth(values, window=50):
    return values.rolling(window=window, min_periods=1).mean()

# 3. Set up the plotting area
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.2)

# --- PLOT 1: Cycle Loss (Structural Correctness) ---
# Combine the two components of cycle loss for each direction
axes[0, 0].plot(df['iteration'], smooth(df['g_A2B_cycA']), label='A→B→A (Cyc A)', color='blue', alpha=0.8)
axes[0, 0].plot(df['iteration'], smooth(df['g_B2A_cycB']), label='B→A→B (Cyc B)', color='cyan', alpha=0.8)
axes[0, 0].set_title('Cycle Consistency Loss\n(Lower = Better Shape Preservation)')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# --- PLOT 2: GAN Loss (Generator Realism) ---
axes[0, 1].plot(df['iteration'], smooth(df['g_A2B_gan']), label='Gen A2B GAN', color='green')
axes[0, 1].plot(df['iteration'], smooth(df['g_B2A_gan']), label='Gen B2A GAN', color='darkgreen')
axes[0, 1].set_title('Generator GAN Loss\n(Stability = Healthy Competition)')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# --- PLOT 3: Adversarial Loss (Discriminator Performance) ---
# This shows how hard it is for the discriminator to tell real from fake
avg_dA = (df['dA_loss_real'] + df['dA_loss_fake']) / 2
avg_dB = (df['dB_loss_real'] + df['dB_loss_fake']) / 2
axes[1, 0].plot(df['iteration'], smooth(avg_dA), label='Disc A Loss', color='red')
axes[1, 0].plot(df['iteration'], smooth(avg_dB), label='Disc B Loss', color='orange')
axes[1, 0].set_title('Discriminator (Adversarial) Loss\n(Lower = Better Discrimination)')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# --- PLOT 4: PSNR (Objective Image Quality) ---
axes[1, 1].plot(df['iteration'], smooth(df['psnr_A']), label='PSNR A', color='purple')
axes[1, 1].plot(df['iteration'], smooth(df['psnr_B']), label='PSNR B', color='magenta')
axes[1, 1].set_title('PSNR Score\n(Higher = Better Reconstruction)')
axes[1, 1].set_ylabel('dB')
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.suptitle(f"CycleGAN Training Metrics Dashboard (Total Iterations: {df['iteration'].max()})", fontsize=16)
plt.savefig('cyclegan_performance_report.png', dpi=300)
plt.show()