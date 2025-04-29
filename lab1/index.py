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

def bezier_point_5th_order(control_points, t):
    """Calculate a point on a 5th order Bezier curve."""
    n = 5 # Order of the curve (degree = n)
    point = np.zeros(2, dtype=np.float64)
    for i in range(n + 1):
        basis = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        point += basis * control_points[i]
    return point

def draw_bezier_curve_5th_order(points, img_shape, num_steps_per_segment=15):
    """Draw the contour using 5th order Bezier curves."""
    # Create a 3-channel white background image for color drawing
    result_img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255
    n_points = len(points)

    if n_points < 6:
        print("Not enough points for 5th order Bezier curves. Need at least 6.")
        # Draw original points if too few
        if n_points > 1:
             cv2.polylines(result_img, [points], isClosed=True, color=(0, 0, 255), thickness=1) # Red color
        return result_img

    all_curve_points = []
    # Generate t values for each segment, excluding endpoint t=1
    t_values = np.linspace(0, 1, num_steps_per_segment, endpoint=False)

    # Iterate through points, taking 6 consecutive points as control points for each segment
    for i in range(n_points):
        # Define control points for the segment using 6 consecutive points (wrap around using modulo)
        control_points = [points[(i + j) % n_points] for j in range(6)]

        # Calculate points along this Bezier curve segment for t in [0, 1)
        for t in t_values:
            curve_pt = bezier_point_5th_order(control_points, t)
            all_curve_points.append(curve_pt.astype(np.int32))

    # Draw the complete contour by connecting all calculated points
    if all_curve_points:
        # Use Red color (BGR format)
        cv2.polylines(result_img, [np.array(all_curve_points)], isClosed=True, color=(0, 0, 255), thickness=1)

    return result_img


# ... rational_bezier_point and draw_rational_bezier_arcs functions remain but are not used ...
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

def draw_rational_bezier_arcs(points, img_shape, weight_middle=0.7, num_steps_per_arc=10):
    """Draw the contour by connecting points calculated along rational Bezier arcs defined by consecutive triplets."""
    # Create a 3-channel white background image to allow for color drawing
    result_img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255
    n_points = len(points)

    if n_points < 3:
        print("Not enough points to draw arcs.")
        # Draw original points if too few
        if n_points > 1:
             # Use blue color (BGR format)
             cv2.polylines(result_img, [points], isClosed=True, color=(255, 0, 0), thickness=1)
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
        # Use blue color (BGR format)
        cv2.polylines(result_img, [np.array(all_arc_points)], isClosed=True, color=(255, 0, 0), thickness=1)

    return result_img
# ... end of unused functions ...


def main():
    # Change image path to shark.png
    image_path = "shark.png"

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

        # print number of points
        print(f"Number of points read from image: {len(points)}")
        # Draw using the 5th order Bezier curve method
        # Experiment with parameters:
        # num_steps_per_segment: How many line segments approximate each curve segment
        result_img = draw_bezier_curve_5th_order(
            points,
            original_img.shape[:2], # Pass (height, width)
            num_steps_per_segment=15 # Number of steps (line segments) per curve segment
        )

        # Display results
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image (Shark)")
        plt.axis('off')
        
        plt.plot(points[:, 0], points[:, 1], 'r.', label='Точки')

        plt.subplot(1, 2, 2)
        # Display the color image
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.plot(points[:, 0], points[:, 1], 'r.', label='Точки')
        plt.title("5th Order Bezier Curve (Red)") # Updated title
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as fnf_error:
         print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()