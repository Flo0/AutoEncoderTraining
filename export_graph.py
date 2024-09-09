from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Load the event data
ea = event_accumulator.EventAccumulator("./runs/Denoising_Unet_MSE_auto_encoder_20240823_170003/events.out.tfevents.1724425203.DESKTOP-V95BOKJ.13004.0")
ea.Reload()

# Get scalars for a specific tag
tag = 'Loss/train'
scalars = ea.Scalars(tag)

# Extract time, step, and value
times = [s.wall_time for s in scalars]
steps = [s.step for s in scalars]
values = [s.value for s in scalars]

# Find the first index where the value is <= 1.0
start_index = next((i for i, v in enumerate(values) if v <= 1.0), None)

# If such an index is found, filter the data
if start_index is not None:
    times = times[start_index:]
    steps = steps[start_index:]
    values = values[start_index:]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot for the normal scale
ax1.plot(steps, values)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss')
ax1.set_title('Denoising Autoencoder, UNET impl, MSE-Loss (Normal Scale)')

# Plot for the log scale
ax2.plot(steps, values)
ax2.set_xlabel('Steps')
ax2.set_ylabel('Loss')
ax2.set_yscale('log')  # Set y-axis to logarithmic scale
ax2.set_title('Denoising Autoencoder, UNET impl, MSE-Loss (Log Scale)')

# Save the plot as an image
plt.tight_layout()
plt.savefig('denoise_mse_unet_plot_with_log_scale_filtered.png')

plt.show()  # Optional: Show the plot if running interactively
