import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_single_line():
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set up the axis limits
    ax.set_xlim([0, 10])  # X-axis range
    ax.set_ylim([0, 10])  # Y-axis range
    ax.set_zlim([0, 10])  # Z-axis range

    # Create a strictly vertical line in the y direction
    x = [5, 5]  # X position (constant for vertical line)
    y = [5, 5]  # Y position (range from 0 to some value, e.g., 7)
    z = [0, 7]  # Z position (constant for vertical line)

    ax.plot(x, y, z, color='blue')

    # Customize the view to emphasize the empty x-z space and vertical y-line
    ax.view_init(elev=20, azim=45)  # Adjust elevation and azimuth angle if needed

    # Remove the grid and the panes (walls)
    # ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Hide the Z-axis completely by setting its limits to zero and turning off ticks
    ax.zaxis.line.set_lw(0)  # Set Z-axis line width to 0 (hide it)
    ax.set_zticks([])  # Remove Z-axis ticks

    # Remove ticks and labels for x and y axes
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('single_line.png')


def plot_gaussian():
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the mean and standard deviation for the normal distribution
    mu_x, mu_y = 5, 5  # Means for x and y
    sigma_x, sigma_y = 1, 1  # Standard deviations for x and y

    # Generate grid data
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    x, y = np.meshgrid(x, y)

    # Calculate the 2D Gaussian distribution values
    z = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
        -(((x - mu_x) ** 2 / (2 * sigma_x ** 2)) + ((y - mu_y) ** 2 / (2 * sigma_y ** 2)))
    )

    # Plot the 3D surface
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

    # Set up the axis limits to match the original plot view
    ax.set_xlim([0, 10])  # X-axis range
    ax.set_ylim([0, 10])  # Y-axis range
    ax.set_zlim([0, np.max(z)])  # Z-axis range

    # Set the view to emphasize the normal distribution shape and match the original view
    ax.view_init(elev=20, azim=45)  # Adjust elevation and azimuth angle

    # Remove the grid and the panes (walls)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Hide the Z-axis ticks
    ax.zaxis.line.set_lw(0)  # Set Z-axis line width to 0 (hide it)
    ax.set_zticks([])  # Remove Z-axis ticks

    # Remove ticks for x and y axes
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('gaussian.png')


plot_single_line()
plot_gaussian()
