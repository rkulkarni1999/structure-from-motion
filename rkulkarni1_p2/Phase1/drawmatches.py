import cv2
import numpy as np

# Assuming pts1 and pts2 are your matched feature points with shape (n, 2)
# and that you have loaded your images into img1 and img2

# Function to draw matches
def draw_matches(img1, pts1, img2, pts2):
    # Create a new output image that concatenates the two images together
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    output_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    output_img[0:h1, 0:w1] = img1
    output_img[0:h2, w1:] = img2
    
    # Draw lines between matching points
    for p1, p2 in zip(pts1, pts2):
        # Generate random color for each line
        # color = np.random.randint(0, 255, (3,)).tolist()
        # Draw points on img1
        for p in pts1:
            cv2.circle(output_img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
            
        # Draw points on img2
        for p in pts2:
            cv2.circle(output_img, (int(p[0] + w1), int(p[1])), 2, (0, 0, 255), -1)
        color = np.random.randint(0, 255, (3,)).tolist()
        color = (0,255,0)
        # Draw the match lines
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0] + w1), int(p2[1]))  # Offset by the width of img1
        cv2.line(output_img, pt1, pt2, color, 1)

    cv2.imshow('Matched Features', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# # Draw the matches (replace these with your actual images and points)
# img1 = cv2.imread('path_to_image1.jpg')  # Load image 1
# img2 = cv2.imread('path_to_image2.jpg')  # Load image 2
# matched_img = draw_matches(img1, pts1, img2, pts2)

# # Display the matched image
# cv2.imshow('Matched Features', matched_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()