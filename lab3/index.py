import turtle
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# --- L-System Fractal (Variant 5: Hexagonal Mosaic) ---

def apply_rules(axiom, rules):
    """Applies L-system rules to the axiom string."""
    result = ""
    for char in axiom:
        result += rules.get(char, char) # Get rule for char, or char itself if no rule
    return result

def generate_l_system(iterations, axiom, rules):
    """Generates the L-system string after a number of iterations."""
    current_string = axiom
    for _ in range(iterations):
        current_string = apply_rules(current_string, rules)
    return current_string

def draw_l_system(t, instructions, angle, distance):
    """Draws the L-system using turtle graphics."""
    stack = []
    screen = t.getscreen()
    screen.tracer(0) # Turn off screen updates for faster drawing

    # Initial position adjustment (optional, centers the drawing better)
    t.penup()
    t.goto(0, -screen.window_height() / 3) # Start lower center
    t.pendown()
    t.setheading(90) # Point turtle upwards initially

    for cmd in instructions:
        if cmd == 'F':
            t.forward(distance)
        elif cmd == '+':
            t.right(angle)
        elif cmd == '-':
            t.left(angle)
        elif cmd == '[':
            position = t.position()
            heading = t.heading()
            stack.append((position, heading))
        elif cmd == ']':
            if stack:
                position, heading = stack.pop()
                t.penup()
                t.goto(position)
                t.setheading(heading)
                t.pendown()
        # Ignore X and Y for drawing

    screen.update() # Update screen once drawing is complete

def run_hexagonal_mosaic(iterations=3, length=10):
    """Sets up and runs the Hexagonal Mosaic L-System."""
    axiom = "X"
    rules = {
        "X": "[-F+F[Y]+F][+F-F[X]-F",
        "Y": "[-F+F[Y]+F][+F-F-F]"
        # F, +, - are constants
    }
    angle = 60

    l_system_string = generate_l_system(iterations, axiom, rules)

    # Setup turtle screen
    screen = turtle.Screen()
    screen.setup(width=800, height=800)
    screen.bgcolor("white")
    screen.title(f"L-System: Hexagonal Mosaic (Iterations: {iterations})")

    # Setup turtle
    t = turtle.Turtle()
    t.speed(0) # Fastest speed
    t.hideturtle()
    t.color("blue")
    t.pensize(1)

    draw_l_system(t, l_system_string, angle, length)

    print(f"Finished drawing L-System. String length: {len(l_system_string)}")
    # Keep window open until clicked
    # screen.exitonclick() # This might interfere if running other fractals later

# --- IFS Fractal (Variant 5: Fern) ---

def run_ifs_fractal(num_points=50000):
    """Generates and plots the Barnsley Fern IFS fractal."""
    print("\n--- IFS Fractal: Barnsley Fern ---")

    # Transformation definitions (a, b, c, d, e, f, p)
    transforms = [
        # T1
        {'a': 0.00, 'b': 0.00, 'd': 0.00, 'e': 0.16, 'c': 0.00, 'f': 0.00, 'p': 0.01},
        # T2
        {'a': 0.85, 'b': 0.04, 'd': -0.04, 'e': 0.85, 'c': 0.00, 'f': 1.60, 'p': 0.85},
        # T3
        {'a': 0.20, 'b': -0.26, 'd': 0.23, 'e': 0.22, 'c': 0.00, 'f': 1.60, 'p': 0.07},
        # T4
        {'a': -0.15, 'b': 0.28, 'd': 0.26, 'e': 0.24, 'c': 0.00, 'f': 0.44, 'p': 0.07}
    ]

    # Extract probabilities and create cumulative distribution
    probabilities = [t['p'] for t in transforms]
    cum_probs = np.cumsum(probabilities)

    # Initialize points array
    points = np.zeros((num_points, 2))
    x, y = 0.0, 0.0 # Start at origin

    start_time = time.time()
    for i in range(1, num_points):
        # Choose transformation based on probability
        rand_val = random.random()
        chosen_transform = None
        for j, p_cum in enumerate(cum_probs):
            if rand_val < p_cum:
                chosen_transform = transforms[j]
                break
        if chosen_transform is None: # Should not happen if probs sum to 1
             chosen_transform = transforms[-1]

        # Apply affine transformation:
        # x_new = a*x + b*y + c
        # y_new = d*x + e*y + f
        x_new = chosen_transform['a'] * x + chosen_transform['b'] * y + chosen_transform['c']
        y_new = chosen_transform['d'] * x + chosen_transform['e'] * y + chosen_transform['f']

        points[i, 0] = x_new
        points[i, 1] = y_new
        x, y = x_new, y_new

    end_time = time.time()
    print(f"Generated {num_points} IFS points in {end_time - start_time:.2f} seconds.")

    # Plotting using Matplotlib
    fig_ifs, ax_ifs = plt.subplots(figsize=(6, 10)) # Fern is taller than wide
    # Use a small point size and green color, no borders
    ax_ifs.scatter(points[:, 0], points[:, 1], s=0.1, color='green', marker='.')
    ax_ifs.set_title("IFS: Barnsley Fern")
    ax_ifs.set_xlabel("X")
    ax_ifs.set_ylabel("Y")
    ax_ifs.set_aspect('equal', adjustable='box') # Ensure correct aspect ratio
    ax_ifs.axis('off') # Hide axes for cleaner look
    fig_ifs.tight_layout()
    # plt.show() # Show separately or integrate later

# --- Algebraic Fractal (Variant 5: Mandelbrot Set) ---

def mandelbrot(c, max_iter):
    """Calculate iterations until divergence for a point c in Mandelbrot set."""
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    # If loop finished because n == max_iter, point is likely in the set (return max_iter)
    # Otherwise, it diverged (return n)
    return n

def run_algebraic_fractal(width=800, height=800, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, max_iter=100):
    """Generates and plots the Mandelbrot set."""
    print("\n--- Algebraic Fractal: Mandelbrot Set ---")

    # Create grid of complex numbers (coordinates)
    real = np.linspace(x_min, x_max, width)
    imag = np.linspace(y_min, y_max, height)
    C = real[:, np.newaxis] + 1j * imag[np.newaxis, :] # Create complex grid

    # Calculate Mandelbrot iterations for each point
    start_time = time.time()
    mandel_map = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            mandel_map[i, j] = mandelbrot(C[i, j], max_iter)
    end_time = time.time()
    print(f"Calculated Mandelbrot set ({width}x{height}, max_iter={max_iter}) in {end_time - start_time:.2f} seconds.")

    # Plotting using Matplotlib
    fig_mandel, ax_mandel = plt.subplots(figsize=(8, 8))
    # Use imshow to display the iteration count as colors
    # Transpose mandel_map because imshow expects (row, col) -> (y, x)
    # Use 'extent' to label axes correctly
    # Use a colormap like 'magma', 'hot', 'twilight_shifted'
    img = ax_mandel.imshow(mandel_map.T, extent=[x_min, x_max, y_min, y_max], cmap='magma', origin='lower')
    ax_mandel.set_title(f"Algebraic: Mandelbrot Set (max_iter={max_iter})")
    ax_mandel.set_xlabel("Re(c)")
    ax_mandel.set_ylabel("Im(c)")
    fig_mandel.colorbar(img, ax=ax_mandel, label='Iterations until divergence')
    fig_mandel.tight_layout()
    # plt.show() # Show separately or integrate later

# --- Main Execution ---
if __name__ == "__main__":
    print("Running Lab 3 Fractals...")

    # B) Run L-System
    print("\n--- L-System: Hexagonal Mosaic ---")
    try:
        # Run turtle graphics in a way that allows subsequent plots
        run_hexagonal_mosaic(iterations=4, length=5) # Adjust iterations/length as needed
        print("L-System window opened. Close it manually to continue.")
        time.sleep(2)
    except turtle.Terminator:
        print("Turtle window closed.")
    except Exception as e:
        print(f"Error running L-System: {e}")

    # A) Run IFS
    run_ifs_fractal()

    # C) Run Algebraic
    run_algebraic_fractal()

    print("\nShowing Matplotlib plots (if any)...")
    plt.show() # Show all matplotlib figures created

    print("\nLab 3 Execution Finished.")
    try:
        print("Click on the L-System window to exit if it's still open.")
        turtle.Screen().exitonclick()
    except turtle.Terminator:
        pass # Window already closed
    except Exception as e:
        print(f"Final turtle exit error (can be ignored if plots shown): {e}")
