import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def plot_depth_and_hist(depth_image, polar_histogram):
        # Display the depth image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(depth_image, cmap='viridis')
        plt.colorbar()
        plt.title('Depth Image')

        # Display the polar histogram
        plt.subplot(1, 2, 2)
        theta = np.linspace(0, 2 * np.pi, len(polar_histogram), endpoint=False)
        bars = plt.bar(theta, polar_histogram, width=(2 * np.pi) / len(polar_histogram), align='edge')
        plt.title('Polar Histogram')
        plt.xlabel('Direction')
        plt.ylabel('Obstacle Density')

        # Adjust the plot
        plt.tight_layout()
        plt.show()

def depth_image_to_polar_histogram(depth_image, num_sectors=8):
    histogram = np.zeros(num_sectors)
    center_x, center_y = depth_image.shape[0] // 2, depth_image.shape[1] // 2
    for x in range(depth_image.shape[0]):
        for y in range(depth_image.shape[1]):
            angle = np.arctan2(y - center_y, x - center_x) + np.pi
            distance = depth_image[x, y]
            sector = int(num_sectors * angle / (2 * np.pi)) % num_sectors
            histogram[sector] += 1 / (distance + 1e-6)
    return histogram

def select_optimal_direction(polar_histogram):
    return np.argmin(polar_histogram)

def calculate_velocity_vector(optimal_sector, vertical_adjustment=2, num_sectors=8, speed=1, preferred_altitude=5, current_altitude=5):
    """
    Calculate a 3D velocity vector based on the optimal sector and vertical obstacle information.
    
    :param optimal_sector: Index of the optimal sector for horizontal movement.
    :param vertical_adjustment: Adjustment needed in the vertical direction (-1 for down, 1 for up, 0 for no change).
    :param num_sectors: Total number of sectors in the polar histogram.
    :param speed: Desired speed of the drone.
    :param preferred_altitude: The drone's preferred cruising altitude.
    :param current_altitude: The drone's current altitude.
    :return: 3D velocity vector (vx, vy, vz).
    """
    # Calculate horizontal components
    angle = (optimal_sector + 0.5) * (2 * np.pi / num_sectors) - np.pi
    vx = np.cos(angle) * speed
    vy = np.sin(angle) * speed
    
    # Calculate vertical component
    vz = 0
    if vertical_adjustment != 0:
        vz = speed * vertical_adjustment
    else:
        # Adjust to return to preferred altitude if no vertical obstacles
        if current_altitude < preferred_altitude:
            vz = speed * 0.5  # Example: ascend with half horizontal speed
        elif current_altitude > preferred_altitude:
            vz = -speed * 0.5  # Example: descend with half horizontal speed

    return vx, vy, vz

def main():
    # Generate a random 8x8 depth image for demonstration
    np.random.seed(42) # For reproducibility
    
    depth_image = np.ones((8, 8)) * 255 # Set the entire image to maximum depth 
    depth_image[2:5, 0:2] = 50
    polar_histogram = depth_image_to_polar_histogram(depth_image)
    optimal_sector = select_optimal_direction(polar_histogram)
    
    plot_depth_and_hist(depth_image, polar_histogram)    
    velocity_vector = calculate_velocity_vector(optimal_sector)
    print(f"Optimal Sector: {optimal_sector}, Velocity Vector: {velocity_vector}")
    print("---")
    
    
if __name__ == "__main__":
    main()
    
    

    
