from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Load the event data
ea = event_accumulator.EventAccumulator("./runs/mnist/vae_20240911-170311/events.out.tfevents.1726066991.DESKTOP-V95BOKJ.7512.0")
ea.Reload()

# Get scalars for a specific tag
tag = 'Loss/Train/KLD'
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
ax1.set_ylabel('KL-D')
ax1.set_title('Fixed-Size VAE, MNIST, KL-Divergenz')

# Plot for the log scale
ax2.plot(steps, values)
ax2.set_xlabel('Steps')
ax2.set_ylabel('KL-D')
ax2.set_yscale('log')  # Set y-axis to logarithmic scale
ax2.set_title('Fixed-Size VAE, MNIST, KL-Divergenz (Log Scale)')

# Save the plot as an image
plt.tight_layout()
plt.savefig('fixed_vae_kld.png')

plt.show()  # Optional: Show the plot if running interactively
