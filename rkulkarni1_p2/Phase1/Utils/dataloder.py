import numpy as np


def features_extraction(data):
    no_of_images = 5
    feature_rgb_values = []
    feature_x = []
    feature_y = []
    feature_flag = []

    # Loop through all matching files
    for n in range(1, no_of_images):
        file_path = f"{data}/matching{n}.txt"
        with open(file_path, "r") as matching_file:
            for i, row in enumerate(matching_file):
                row_elements = row.strip().split()
                
                if i == 0:
                    # First row contains the number of features, which is not directly used here
                    continue
                
                # Extracting values from the row
                columns = [float(x) for x in row_elements]
                r_value, g_value, b_value = columns[1:4]
                feature_rgb_values.append([r_value, g_value, b_value])

                # Preparing x, y, and flag rows for this feature across all images
                x_row = np.zeros(no_of_images)
                y_row = np.zeros(no_of_images)
                flag_row = np.zeros(no_of_images, dtype=int)
                
                nMatches = int(columns[0])
                current_x, current_y = columns[4:6]
                
                # Process the current image's feature
                x_row[n-1] = current_x
                y_row[n-1] = current_y
                flag_row[n-1] = 1
                
                # Process matched features in other images
                for m in range(nMatches - 1):
                    idx = 6 + m * 3
                    image_id, image_id_x, image_id_y = int(columns[idx]), columns[idx + 1], columns[idx + 2]
                    x_row[image_id - 1] = image_id_x
                    y_row[image_id - 1] = image_id_y
                    flag_row[image_id - 1] = 1
                
                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)

    # Convert lists to NumPy arrays
    feature_x = np.array(feature_x)
    feature_y = np.array(feature_y)
    feature_flag = np.array(feature_flag)
    feature_rgb_values = np.array(feature_rgb_values)
    return feature_x, feature_y, feature_flag, feature_rgb_values