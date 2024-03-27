import numpy as np
import matplotlib.pyplot as plt

def plotxz(points):
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4']

    # Plotting
    plt.figure(figsize=(10, 8))  # Set figure size

    # Loop through each set of points

    for idx, X in enumerate(points):
        valid_points = X[X[:, 2] > 0]

        if valid_points.size > 0:
            x_coords = valid_points[:, 0]  # X coordinates
            z_coords = valid_points[:, 2]  # Z coordinates
            plt.scatter(x_coords, z_coords,s=10, color=colors[idx], label=labels[idx])

    plt.title('Projection of 3D Points Sets on the X-Z Plane')  # Set title
    plt.xlabel('X')  # Set X-axis label
    plt.ylabel('Z')  # Set Z-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    plt.show()  # Display the plo