import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def read_points_from_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to create binary image
    # Use THRESH_BINARY_INV if the object is black on white background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) 
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Ensure points are in shape (N, 2)
        return largest_contour.reshape(-1, 2)
    else:
        return np.array([])

def rational_bezier_point(points, weights, t):
    """Calculate a point on a rational Bezier curve"""
    n = len(points) - 1
    numerator = np.zeros(2, dtype=np.float64) # Use float64 for precision
    denominator = 0.0
    
    for i in range(n + 1):
        basis = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        numerator += basis * weights[i] * points[i]
        denominator += basis * weights[i]
    
    # Avoid division by zero
    if denominator == 0:
        # Return the last control point or an average if t=1 and weights cause issues
        return points[-1] if t == 1 else points[0] 
    
    return numerator / denominator

# This function is no longer used in the revised approach
# def fit_rational_elliptical_arc(points, weight_middle=1.5):
#     ...

# Replace the previous drawing function with one that draws arcs
def draw_rational_bezier_arcs(points, img_shape, weight_middle=0.7, num_steps_per_arc=10):
    # "Draw the contour by connecting points calculated along rational Bezier arcs defined by consecutive triplets.\"\"\"
    result_img = np.ones(img_shape, dtype=np.uint8) * 255
    n_points = len(points)

    if n_points < 3:
        print("Not enough points to draw arcs.")
        # Draw original points if too few
        if n_points > 1:
             cv2.polylines(result_img, [points], isClosed=True, color=(0, 0, 0), thickness=1)
        return result_img

    all_arc_points = []
    weights = [1.0, weight_middle, 1.0]
    # Generate t values for each arc segment, excluding the endpoint t=1
    # because it will be the start point t=0 of the next segment.
    t_values = np.linspace(0, 1, num_steps_per_arc, endpoint=False)

    for i in range(n_points):
        # Define control points for the arc segment using 3 consecutive points
        p0 = points[i]
        p1 = points[(i + 1) % n_points]
        p2 = points[(i + 2) % n_points]
        control_points = [p0, p1, p2]

        # Calculate points along this arc segment for t in [0, 1)
        for t in t_values:
            arc_pt = rational_bezier_point(control_points, weights, t)
            # Ensure points are integers for drawing
            all_arc_points.append(arc_pt.astype(np.int32))

    # Draw the complete contour by connecting all calculated points from all arcs
    if all_arc_points:
        cv2.polylines(result_img, [np.array(all_arc_points)], isClosed=True, color=(0, 0, 0), thickness=1)

    return result_img


def main():
    # Use relative path within the lab1 directory
    image_path = "ostritch.png" 
    
    try:
        # Read original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}. Make sure it's in the same directory as the script or provide the correct path.")
        
        # Read points from image
        points = read_points_from_image(image_path)
        
        if len(points) == 0:
            print("No contours found in the image.")
            return

        # Draw using the rational Bezier arc method
        # Experiment with parameters:
        # weight_middle: Affects curve shape (0.5=parabolic, <1 pulls away, >1 pulls towards P1)
        # num_steps_per_arc: How many line segments approximate each arc
        result_img = draw_rational_bezier_arcs(
            points,
            original_img.shape[:2], # Pass (height, width)
            weight_middle=0.7,      # Weight for middle control point (adjust for shape)
            num_steps_per_arc=10    # Number of steps (line segments) per arc
        )

        # Display results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_img, cmap='gray')
        plt.title("Rational Bezier Arcs") # Updated title
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError as fnf_error:
         print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
