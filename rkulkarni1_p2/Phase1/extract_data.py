import os
import numpy as np

def process_dataset(folder_path):
    # Initialize lists to store features
    feature_rgb_values = []
    feature_x = []
    feature_y = []
    feature_flag = []
    
    # Assume we have 5 images based on the previous function, adjust as needed
    no_of_images = 5

    all_files = os.listdir(folder_path)
    
    for file in all_files:
        # Check if the file is a matching file based on its name
        if file.endswith('.txt') and file.startswith("matching"):
            file_path = os.path.join(folder_path, file)
            
            with open(file_path, "r") as matching_file:
                for i, row in enumerate(matching_file):
                    if i == 0:  # First row having nFeatures/no of features
                        nfeatures = int(row.split(':')[1])
                    else:
                        # Initialize arrays for this row
                        x_row = np.zeros((1, no_of_images))
                        y_row = np.zeros((1, no_of_images))
                        flag_row = np.zeros((1, no_of_images), dtype=int)
                        row_elements = row.split()
                        columns = np.asarray([float(x) for x in row_elements])

                        nMatches = columns[0]
                        r_value = columns[1]
                        g_value = columns[2]
                        b_value = columns[3]

                        feature_rgb_values.append([r_value, g_value, b_value])
                        current_x = columns[4]
                        current_y = columns[5]

                        # Assuming 'n' is derived from file name for x_row, y_row, flag_row index
                        n = int(file.replace("matching", "").replace(".txt", "")) - 1
                        x_row[0, n] = current_x
                        y_row[0, n] = current_y
                        flag_row[0, n] = 1

                        m = 1
                        while nMatches > 1:
                            image_id = int(columns[5 + m])
                            image_id_x = columns[6 + m]
                            image_id_y = columns[7 + m]
                            m += 3
                            nMatches -= 1

                            x_row[0, image_id - 1] = image_id_x
                            y_row[0, image_id - 1] = image_id_y
                            flag_row[0, image_id - 1] = 1

                        feature_x.append(x_row)
                        feature_y.append(y_row)
                        feature_flag.append(flag_row)

    # Convert lists to NumPy arrays and reshape
    feature_x = np.asarray(feature_x).reshape(-1, no_of_images)
    feature_y = np.asarray(feature_y).reshape(-1, no_of_images)
    feature_flag = np.asarray(feature_flag).reshape(-1, no_of_images)
    feature_rgb_values = np.asarray(feature_rgb_values).reshape(-1, 3)

    return feature_x, feature_y, feature_flag, feature_rgb_values

# def features_extraction(data):

#     no_of_images = 5
#     feature_rgb_values = []
#     feature_x = []
#     feature_y = []

#     "We have 4 matching.txt files"
#     feature_flag = []

#     for n in range(1, no_of_images):
#         file = data + "/matching" + str(n) + ".txt"
#         matching_file = open(file,"r")
#         nfeatures = 0

#         for i, row in enumerate(matching_file):
#             if i == 0:  #1st row having nFeatures/no of features
#                 row_elements = row.split(':')
#                 nfeatures = int(row_elements[1])
#             else:
#                 x_row = np.zeros((1,no_of_images))
#                 y_row = np.zeros((1,no_of_images))
#                 flag_row = np.zeros((1,no_of_images), dtype = int)
#                 row_elements = row.split()
#                 columns = [float(x) for x in row_elements]
#                 columns = np.asarray(columns)

#                 nMatches = columns[0]
#                 r_value = columns[1]
#                 b_value = columns[2]
#                 g_value = columns[3]

#                 feature_rgb_values.append([r_value,g_value,b_value])
#                 current_x = columns[4]
#                 current_y = columns[5]

#                 x_row[0,n-1] = current_x
#                 y_row[0,n-1] = current_y
#                 flag_row[0,n-1] = 1

#                 m = 1
#                 while nMatches > 1:
#                     image_id = int(columns[5+m])
#                     image_id_x = int(columns[6+m])
#                     image_id_y = int(columns[7+m])
#                     m = m+3
#                     nMatches = nMatches - 1

#                     x_row[0, image_id - 1] = image_id_x
#                     y_row[0, image_id - 1] = image_id_y
#                     flag_row[0, image_id - 1] = 1

#                 feature_x.append(x_row)
#                 feature_y.append(y_row)
#                 feature_flag.append(flag_row)

#     feature_x = np.asarray(feature_x).reshape(-1,no_of_images)
#     feature_y = np.asarray(feature_y).reshape(-1,no_of_images)
#     feature_flag = np.asarray(feature_flag).reshape(-1,no_of_images)
#     feature_rgb_values = np.asarray(feature_rgb_values).reshape(-1,3)

#     return feature_x, feature_y, feature_flag, feature_rgb_values


# data = "rkulkarni1_p2\Phase1\P3Data\P3Data"
# feature_x, feature_y, feature_flag, feature_rgb_values = process_dataset(data)
# print(feature_x[9], feature_y[9], feature_flag[9], feature_rgb_values[9])