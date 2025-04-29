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

def draw_rational_bezier_approximation(points, img_shape, num_points_per_segment=7, weight_middle=0.8):
    """Draw the contour by approximating points using rational Bezier curves."""
    result_img = np.ones(img_shape, dtype=np.uint8) * 255
    
    n_points = len(points)
    if n_points < num_points_per_segment:
        print("Not enough points to draw.")
        # Draw original points if too few
        if n_points > 1:
             cv2.polylines(result_img, [points], isClosed=True, color=(0, 0, 0), thickness=1)
        return result_img

    smoothed_points = []
    
    # Iterate through each original point to calculate its smoothed position
    for i in range(n_points):
        # Define the segment centered around point i (using modulo for wrap-around)
        # Ensure indices are within bounds using modulo operator
        segment_indices = [(i + j - num_points_per_segment // 2 + n_points) % n_points for j in range(num_points_per_segment)]
        segment = points[segment_indices]

        # Use start, middle, and end points of the segment as control points
        control_points = [segment[0], segment[num_points_per_segment // 2], segment[-1]]
        weights = [1.0, weight_middle, 1.0]

        # Calculate the point on the Bezier curve corresponding to the center (t=0.5)
        # This point represents the smoothed position for points[i]
        smoothed_pt = rational_bezier_point(control_points, weights, 0.5) 
        smoothed_points.append(smoothed_pt.astype(np.int32))

    # Draw a closed polyline through the smoothed points
    if smoothed_points:
        cv2.polylines(result_img, [np.array(smoothed_points)], isClosed=True, color=(0, 0, 0), thickness=1)

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
        
        # Draw using the rational Bezier approximation method
        # Experiment with parameters:
        # num_points_per_segment: Size of the window for smoothing (odd number recommended)
        # weight_middle: Affects curve shape (0.5=parabolic, <1 pulls away, >1 pulls towards P1)
        result_img = draw_rational_bezier_approximation(
            points, 
            original_img.shape[:2], # Pass (height, width)
            num_points_per_segment=7, # Window size
            weight_middle=0.7       # Weight for middle control point (adjust for desired smoothness/shape)
        )
        
        # Display results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_img, cmap='gray')
        plt.title("Rational Bezier Approximation") # Updated title
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError as fnf_error:
         print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
