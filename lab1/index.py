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

def draw_points(points, img_shape, color=(0, 0, 255), radius=2):
    """Draws the contour points as small circles."""
    result_img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255
    if len(points) == 0:
        return result_img
    for pt in points:
        cv2.circle(result_img, tuple(pt), radius, color, -1) # Use BGR color
    return result_img

def draw_bezier_curve_5th_order_full(points, img_shape, color=(255, 0, 0), thickness=1, num_steps_per_segment=15):
    """Draw the full contour using 5th order Bezier curves."""
    result_img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255
    n_points = len(points)

    if n_points < 6:
        print("Not enough points for 5th order Bezier curves. Need at least 6.")
        if n_points > 1:
             cv2.polylines(result_img, [points], isClosed=True, color=color, thickness=thickness)
        return result_img

    all_curve_points = []
    t_values = np.linspace(0, 1, num_steps_per_segment, endpoint=False)

    for i in range(n_points):
        control_points = [points[(i + j) % n_points] for j in range(6)]
        for t in t_values:
            curve_pt = bezier_point_5th_order(control_points, t)
            all_curve_points.append(curve_pt.astype(np.int32))

    if all_curve_points:
        cv2.polylines(result_img, [np.array(all_curve_points)], isClosed=True, color=color, thickness=thickness)

    return result_img

# --- Modified Simplified Function ---
def draw_bezier_curve_5th_order_simplified(simplified_points, img_shape, curve_color=(255, 0, 0), point_color=(0, 0, 255), thickness=1, point_radius=3, num_steps_per_segment=15):
    """
    Draws a 5th order Bezier curve based on a *simplified* set of points
    and highlights only those simplified points.
    """
    result_img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255
    n_points = len(simplified_points)

    if n_points < 6:
        print("Warning: Not enough simplified points for 5th order Bezier curves (need >= 6). Drawing polyline.")
        if n_points > 1:
             cv2.polylines(result_img, [simplified_points], isClosed=True, color=curve_color, thickness=thickness)
        # Draw the simplified points anyway
        for pt in simplified_points:
            cv2.circle(result_img, tuple(pt), point_radius, point_color, -1)
        return result_img

    all_curve_points = []
    t_values = np.linspace(0, 1, num_steps_per_segment, endpoint=False) # endpoint=False avoids duplicating points

    # Generate Bezier curve segments using the simplified points as control points
    for i in range(n_points):
        # Define the 6 control points for the segment starting at simplified_points[i]
        control_points = [simplified_points[(i + j) % n_points] for j in range(6)]
        for t in t_values:
            curve_pt = bezier_point_5th_order(control_points, t)
            all_curve_points.append(curve_pt.astype(np.int32))

    # Draw the curve connecting the calculated points
    if all_curve_points:
        # Ensure the array is contiguous and correctly shaped for polylines
        curve_np = np.array(all_curve_points).reshape(-1, 1, 2)
        cv2.polylines(result_img, [curve_np], isClosed=True, color=curve_color, thickness=thickness)

    # Draw *only* the simplified points used as the basis
    for pt in simplified_points:
        cv2.circle(result_img, tuple(pt), point_radius, point_color, -1) # BGR

    return result_img

def main():
    image_path = "shark.png"
    # --- Parameters ---
    num_steps_per_segment = 15 # Steps for curve smoothness
    # Epsilon for approxPolyDP: Higher value -> fewer points (more simplification)
    # Adjust this value to change the number of points in the simplified view
    epsilon_factor = 0.0003 # Percentage of contour perimeter

    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}.")

        points = read_points_from_image(image_path)
        if len(points) == 0:
            print("No contours found in the image.")
            return

        img_shape = original_img.shape[:2]

        # --- Simplify Contour ---
        perimeter = cv2.arcLength(points.reshape(-1, 1, 2), True)
        epsilon = epsilon_factor * perimeter
        simplified_points = cv2.approxPolyDP(points.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
        print(f"Original points: {len(points)}, Simplified points: {len(simplified_points)}")


        # --- Generate the different views ---
        # Panel 2: Original dense points
        img_points_only = draw_points(points, img_shape, color=(0, 0, 255)) # Red points

        # Panel 3: Full Bezier curve using original points
        img_bezier_full = draw_bezier_curve_5th_order_full(
            points, # Use original points
            img_shape,
            color=(255, 0, 0), # Blue curve
            num_steps_per_segment=num_steps_per_segment
        )

        # Panel 4: Simplified Bezier curve using simplified points
        img_bezier_simplified = draw_bezier_curve_5th_order_simplified(
            simplified_points, # Use simplified points
            img_shape,
            curve_color=(255, 0, 0), # Blue curve
            point_color=(0, 0, 255), # Red points (only simplified ones)
            num_steps_per_segment=num_steps_per_segment
        )

        # --- Display results in a 2x2 grid ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Моделювання контура кривими Безьє 5-го порядку', fontsize=16)

        # Panel 1: Original Image
        axs[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Вхідний малюнок (Акула)")
        axs[0, 0].axis('off')

        # Panel 2: Found Contour Points (Original Dense)
        axs[0, 1].imshow(cv2.cvtColor(img_points_only, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title(f"Знайдені контури ({len(points)} точок)")
        axs[0, 1].axis('off')

        # Panel 3: Full Bezier Curve (Based on Original Points)
        axs[1, 0].imshow(cv2.cvtColor(img_bezier_full, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title("Повний контур (Безьє 5-го порядку)")
        axs[1, 0].axis('off')

        # Panel 4: Simplified Bezier Curve + Simplified Points
        axs[1, 1].imshow(cv2.cvtColor(img_bezier_simplified, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title(f"Спрощений контур ({len(simplified_points)} точок)")
        axs[1, 1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plt.show()

    except FileNotFoundError as fnf_error:
         print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
