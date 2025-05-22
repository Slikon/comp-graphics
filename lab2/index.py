# /Users/vkulyk/Desktop/Study/comp_graphics/kpi-graphics/lab2/index.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from matplotlib.widgets import Slider  # Import Slider

# --- Parameters ---
R_BASE = 1.0  # Base radius for the sphere interpretation
R_MOD = 0.3   # Modifier radius R from the formula (RE-ADDED)
U_RES = 50    # Resolution for u parameter (longitude)
V_RES = 50    # Resolution for v parameter (latitude)
SHARK_IMAGE_PATH = "shark.png"  # Relative path to shark image in lab2 folder

# --- Surface Definition (Variant 5: Місяць) --- REVERTED TO ORIGINAL FORMULA
def moon_surface(u, v, R_base, R_mod):
    """
    Calculates the coordinates of the 'Moon' surface based on the formula:
    x = x_ш + R_mod * (v / 45°)^2
    y = y_ш
    z = 2 * z_ш
    where (x_ш, y_ш, z_ш) are points on a base sphere (radius R_base).
    u in [0, 2*pi], v in [0, pi]
    """
    # Base sphere coordinates
    x_sh = R_base * np.sin(v) * np.cos(u)
    y_sh = R_base * np.sin(v) * np.sin(u)
    z_sh = R_base * np.cos(v)

    # Apply the transformations from the formula
    angle_45_rad = np.pi / 4
    x = x_sh + R_mod * (v / angle_45_rad)**2
    y = y_sh
    z = 2 * z_sh

    return x, y, z

# --- Euclidean Transformations ---
def rotate_x(x, y, z, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    y_new = y * cos_a - z * sin_a
    z_new = y * sin_a + z * cos_a
    return x, y_new, z_new

def rotate_y(x, y, z, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_new = x * cos_a + z * sin_a
    z_new = -x * sin_a + z * cos_a
    return x_new, y, z_new

def rotate_z(x, y, z, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a
    return x_new, y_new, z

def translate(x, y, z, tx, ty, tz):
    return x + tx, y + ty, z + tz

def read_points_from_image(image_path):
    """Reads the largest contour points from an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour.reshape(-1, 2).astype(float)
    else:
        return np.array([])

# --- Function to map 2D points to UV space ---
def map_contour_to_uv(points_2d, u_range=(-np.pi/2, np.pi/2), v_range=(-np.pi/2, np.pi/2), u_offset=0, v_offset=0, u_scale=1.0, v_scale=1.0):
    """Normalizes 2D points and maps them to UV space with offset and scale."""
    if points_2d.size == 0:
        return np.array([]), np.array([])

    min_coords = points_2d.min(axis=0)
    max_coords = points_2d.max(axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1
    normalized_points = (points_2d - min_coords) / range_coords

    u_min, u_max = u_range
    v_min, v_max = v_range
    u_span = u_max - u_min
    v_span = v_max - v_min

    # v_mapped = v_min + (normalized_points[:, 1] * v_span * v_scale)
    v_mapped = v_min + (normalized_points[:, 1] * v_span * v_scale) + v_offset
    # u_mapped = u_min + (normalized_points[:, 0] * u_span * u_scale)
    u_mapped = u_min + (normalized_points[:, 0] * u_span * u_scale) + u_offset

    u_mapped = np.mod(u_mapped, u_max)
    v_mapped = np.clip(v_mapped, v_min, v_max)

    return u_mapped, v_mapped

# --- Global variables for plot elements ---
ax = None
surf_plot = None
shark_plot = None
x_surf, y_surf, z_surf = None, None, None
u_grid, v_grid = None, None
shark_points_2d = None

# --- Update Function for Sliders ---
def update(val):
    global surf_plot, shark_plot

    angle_x = slider_rot_x.val
    angle_y = slider_rot_y.val
    angle_z = slider_rot_z.val
    trans_x = slider_trans_x.val
    trans_y = slider_trans_y.val
    trans_z = slider_trans_z.val
    shark_u_offset = slider_shark_u_off.val
    shark_v_offset = slider_shark_v_off.val
    shark_u_scale = slider_shark_u_scl.val
    shark_v_scale = slider_shark_v_scl.val

    x_transformed, y_transformed, z_transformed = rotate_x(x_surf, y_surf, z_surf, angle_x)
    x_transformed, y_transformed, z_transformed = rotate_y(x_transformed, y_transformed, z_transformed, angle_y)
    x_transformed, y_transformed, z_transformed = rotate_z(x_transformed, y_transformed, z_transformed, angle_z)
    x_transformed, y_transformed, z_transformed = translate(x_transformed, y_transformed, z_transformed, trans_x, trans_y, trans_z)

    x_shark_transformed, y_shark_transformed, z_shark_transformed = [], [], []
    if shark_points_2d is not None and shark_points_2d.size > 0:
        u_shark, v_shark = map_contour_to_uv(
            shark_points_2d,
            u_range=(0, 2*np.pi),  # Reverted U range
            v_range=(0, np.pi),    # Reverted V range
            u_offset=shark_u_offset,
            v_offset=shark_v_offset,
            u_scale=shark_u_scale,
            v_scale=shark_v_scale
        )
        x_shark, y_shark, z_shark = moon_surface(u_shark, v_shark, R_BASE, R_MOD)

        x_shark_transformed, y_shark_transformed, z_shark_transformed = rotate_x(x_shark, y_shark, z_shark, angle_x)
        x_shark_transformed, y_shark_transformed, z_shark_transformed = rotate_y(x_shark_transformed, y_shark_transformed, z_shark_transformed, angle_y)
        x_shark_transformed, y_shark_transformed, z_shark_transformed = rotate_z(x_shark_transformed, y_shark_transformed, z_shark_transformed, angle_z)
        x_shark_transformed, y_shark_transformed, z_shark_transformed = translate(x_shark_transformed, y_shark_transformed, z_shark_transformed, trans_x, trans_y, trans_z)

    if surf_plot is not None:
        surf_plot.remove()
    if shark_plot is not None and len(shark_plot) > 0:
        if hasattr(shark_plot, '__iter__'):
            line = shark_plot.pop(0)
            line.remove()
        else:
            shark_plot.remove()
            shark_plot = None

    surf_plot = ax.plot_surface(x_transformed, y_transformed, z_transformed,
                                # color='#073B4C',  # Dark blue
                                # color='#D0D7DA',  # white blue
                                color='#FFFFFF',  # white blue
                                edgecolor='k', lw=0.2, rstride=2, cstride=2, alpha=0.4)

    if len(x_shark_transformed) > 0:
        shark_plot = ax.plot(np.append(x_shark_transformed, x_shark_transformed[0]),
                             np.append(y_shark_transformed, y_shark_transformed[0]),
                             np.append(z_shark_transformed, z_shark_transformed[0]),
                             color='red', lw=5, label='Shark Contour')
    else:
        shark_plot = None

    plt.draw()

# --- Main Function ---
def main():
    global ax, surf_plot, shark_plot, x_surf, y_surf, z_surf, u_grid, v_grid, shark_points_2d
    global slider_rot_x, slider_rot_y, slider_rot_z, slider_trans_x, slider_trans_y, slider_trans_z
    global slider_shark_u_off, slider_shark_v_off, slider_shark_u_scl, slider_shark_v_scl

    u = np.linspace(0, 2 * np.pi, U_RES)  # Reverted U range
    v = np.linspace(0, np.pi, V_RES)      # Reverted V range
    u_grid, v_grid = np.meshgrid(u, v)

    x_surf, y_surf, z_surf = moon_surface(u_grid, v_grid, R_BASE, R_MOD)

    try:
        shark_points_2d = read_points_from_image(SHARK_IMAGE_PATH)
        if shark_points_2d.size == 0:
            print(f"Warning: No contour found in {SHARK_IMAGE_PATH}")
            shark_points_2d = None
    except FileNotFoundError as e:
        print(e)
        shark_points_2d = None
    except ImportError:
        print("Error: cv2 library not found. Please install opencv-python: pip install opencv-python")
        shark_points_2d = None
    except Exception as e:
        print(f"An unexpected error occurred during shark loading: {e}")
        shark_points_2d = None

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d', position=[0.05, 0.3, 0.9, 0.65])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Поверхня "Місяць" з контуром Акули (Інтерактивна)')

    init_angle_x, init_angle_y, init_angle_z = 0, 0, 0
    init_trans_x, init_trans_y, init_trans_z = 0, 0, 0
    init_shark_u_offset = np.pi / 2  # Reverted initial U offset
    init_shark_v_offset = np.pi / 4  # Reverted initial V offset
    init_shark_u_scale = 0.5
    init_shark_v_scale = 0.5

    axcolor = 'lightgoldenrodyellow'
    slider_h = 0.02
    slider_w = 0.35
    row1_y = 0.18
    row2_y = row1_y - slider_h - 0.01
    row3_y = row2_y - slider_h - 0.01
    row4_y = row3_y - slider_h - 0.01
    col1_x = 0.1
    col2_x = 0.55

    ax_rot_x = plt.axes([col1_x, row1_y, slider_w, slider_h], facecolor=axcolor)
    ax_rot_y = plt.axes([col2_x, row1_y, slider_w, slider_h], facecolor=axcolor)
    slider_rot_x = Slider(ax_rot_x, 'Rot X (°)', -180, 180, valinit=init_angle_x)
    slider_rot_y = Slider(ax_rot_y, 'Rot Y (°)', -180, 180, valinit=init_angle_y)

    ax_rot_z = plt.axes([col1_x, row2_y, slider_w, slider_h], facecolor=axcolor)
    ax_trans_x = plt.axes([col2_x, row2_y, slider_w, slider_h], facecolor=axcolor)
    slider_rot_z = Slider(ax_rot_z, 'Rot Z (°)', -180, 180, valinit=init_angle_z)
    slider_trans_x = Slider(ax_trans_x, 'Trans X', -2*R_BASE, 2*R_BASE, valinit=init_trans_x)

    ax_trans_y = plt.axes([col1_x, row3_y, slider_w, slider_h], facecolor=axcolor)
    ax_trans_z = plt.axes([col2_x, row3_y, slider_w, slider_h], facecolor=axcolor)
    slider_trans_y = Slider(ax_trans_y, 'Trans Y', -2*R_BASE, 2*R_BASE, valinit=init_trans_y)
    slider_trans_z = Slider(ax_trans_z, 'Trans Z', -2*R_BASE, 2*R_BASE, valinit=init_trans_z)

    ax_shark_u_off = plt.axes([col1_x, row4_y, slider_w/2 - 0.01, slider_h], facecolor=axcolor)
    ax_shark_v_off = plt.axes([col1_x + slider_w/2, row4_y, slider_w/2 - 0.01, slider_h], facecolor=axcolor)
    ax_shark_u_scl = plt.axes([col2_x, row4_y, slider_w/2 - 0.01, slider_h], facecolor=axcolor)
    ax_shark_v_scl = plt.axes([col2_x + slider_w/2, row4_y, slider_w/2 - 0.01, slider_h], facecolor=axcolor)
    slider_shark_u_off = Slider(ax_shark_u_off, 'Shark U Off', 0, 2*np.pi, valinit=init_shark_u_offset)  # Reverted range
    slider_shark_v_off = Slider(ax_shark_v_off, 'Shark V Off', 0, np.pi, valinit=init_shark_v_offset)  # Reverted range
    slider_shark_u_scl = Slider(ax_shark_u_scl, 'Shark U Scale', 0.1, 2.0, valinit=init_shark_u_scale)
    slider_shark_v_scl = Slider(ax_shark_v_scl, 'Shark V Scale', 0.1, 2.0, valinit=init_shark_v_scale)

    slider_rot_x.on_changed(update)
    slider_rot_y.on_changed(update)
    slider_rot_z.on_changed(update)
    slider_trans_x.on_changed(update)
    slider_trans_y.on_changed(update)
    slider_trans_z.on_changed(update)
    slider_shark_u_off.on_changed(update)
    slider_shark_v_off.on_changed(update)
    slider_shark_u_scl.on_changed(update)
    slider_shark_v_scl.on_changed(update)

    update(None)

    try:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        z_lim = ax.get_zlim()
        max_range = np.array([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]]).max() / 2.0
        mid_x = np.mean(x_lim)
        mid_y = np.mean(y_lim)
        mid_z = np.mean(z_lim)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    except Exception as e:
        print(f"Warning: Could not set plot limits automatically. {e}")

    plt.show()

if __name__ == "__main__":
    main()