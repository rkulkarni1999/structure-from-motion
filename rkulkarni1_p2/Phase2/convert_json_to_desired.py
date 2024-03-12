import json
from math import atan, pi

# Function to calculate camera angle x
def calculate_camera_angle_x(w, fl_x):
    return 2 * atan(w / (2 * fl_x))

# Load the input JSON file
input_file_path = './rkulkarni1_p2/Phase2/outputs/custom_dataset/mouse_nf_format/transforms.json'  # Update this path if the file is in a different location
with open(input_file_path, 'r') as infile:
    input_json = json.load(infile)

# Calculate camera_angle_x
w = input_json['w']
fl_x = input_json['fl_x']
camera_angle_x = calculate_camera_angle_x(w, fl_x)

# Prepare the output JSON structure
output_json = {
    "camera_angle_x": camera_angle_x,
    "frames": []
}

# Transform each frame to match the specified format using colmap_im_id for naming
for frame in input_json['frames']:
    colmap_im_id = frame['colmap_im_id']
    new_frame = {
        "file_path": f"./train/r_{colmap_im_id}",
        "transform_matrix": frame['transform_matrix']
        # Note: The 'rotation' value calculation and inclusion are omitted due to lack of specific instructions
    }
    output_json['frames'].append(new_frame)

# Save the transformed data to an output JSON file
output_file_path = './rkulkarni1_p2/Phase2/outputs/custom_dataset/mouse_nf_format/transforms_mouse.json'
with open(output_file_path, 'w') as outfile:
    json.dump(output_json, outfile, indent=4)

print(f"Transformed JSON saved to {output_file_path}")
