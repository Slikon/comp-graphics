# Lab 2: 3D Surface Generation - Questions and Answers

## Surface Theory and Implementation

### Q1: What type of surface did you implement and how is it defined mathematically?
**A:** I implemented a "Moon" surface (Variant 5) which is a modified spherical surface. Mathematically, it's defined by transforming the points of a base sphere using the formula:
```
x = x_sphere + R_mod * (v / 45°)²
y = y_sphere
z = 2 * z_sphere
```
Where:
- (x_sphere, y_sphere, z_sphere) are points on a base sphere with radius R_BASE
- R_mod is a modifier radius that controls the "bulge" of the moon
- v is the latitude parameter in radians
- The formula creates a asymmetrically deformed sphere with greater extension in the z dimension and a progressive deformation in x based on latitude

### Q2: Explain the parameterization of your surface. What do u and v represent?
**A:** The surface is parameterized using two parameters:
- u ∈ [0, 2π] represents the longitude (angle around the equator)
- v ∈ [0, π] represents the latitude (angle from north to south pole)

This is a standard spherical parameterization where:
- u = 0 to 2π sweeps around the sphere horizontally
- v = 0 represents the north pole, v = π/2 is the equator, and v = π is the south pole
- Together, these parameters define unique points on the surface

### Q3: Describe the Euclidean transformations you implemented and their mathematical basis.
**A:** I implemented three types of Euclidean transformations:

1. **Rotation** around x, y, and z axes:
   - For x-axis rotation: [x, y', z'] = [x, y·cos(α) - z·sin(α), y·sin(α) + z·cos(α)]
   - For y-axis rotation: [x', y, z'] = [x·cos(α) + z·sin(α), y, -x·sin(α) + z·cos(α)]
   - For z-axis rotation: [x', y', z] = [x·cos(α) - y·sin(α), x·sin(α) + y·cos(α), z]
   
   Where α is the rotation angle in radians.

2. **Translation** along x, y, and z axes:
   - [x', y', z'] = [x + tx, y + ty, z + tz]
   
   Where (tx, ty, tz) are the translation distances.

These transformations preserve the shape and size of the object while changing its position and orientation in 3D space.

### Q4: How did you map the 2D shark contour onto the 3D surface?
**A:** I mapped the 2D shark contour onto the 3D surface through these steps:
1. Read the 2D contour points from the shark image using OpenCV
2. Normalize the 2D points to a suitable range
3. Map these normalized points to parameters (u,v) in the domain of the surface:
   - x-coordinates → u parameter (longitude)
   - y-coordinates → v parameter (latitude)
4. Apply offset and scaling factors to control the placement and size
5. Calculate the corresponding 3D surface points using the moon_surface function
6. Render these 3D points as a connected line on the surface

The mapping function (`map_contour_to_uv`) includes parameters for offset and scale in both directions, allowing fine control over the placement.

## Technical Implementation

### Q5: Explain how your interactive controls work and what they allow the user to do.
**A:** The interactive controls use Matplotlib's Slider widgets to allow real-time manipulation:

1. **Rotation sliders** (Rot X, Rot Y, Rot Z): Control rotation around each axis from -180° to 180°
2. **Translation sliders** (Trans X, Trans Y, Trans Z): Move the surface along each axis
3. **Shark mapping sliders**:
   - U/V Offset: Change the position of the shark contour on the surface
   - U/V Scale: Control the size and aspect ratio of the shark contour

The controls call an update function that recalculates and redraws the surface and contour whenever a slider value changes. This enables exploring different perspectives and placements without rerunning the program.

### Q6: What challenges did you face when implementing this 3D surface and how did you overcome them?
**A:** Several challenges arose during implementation:

1. **Parameter continuity**: Ensuring the surface was continuous at u=0 and u=2π boundary required careful handling of the parameter ranges.

2. **Contour mapping**: Finding the right mapping from 2D to 3D space required experimentation with different offset and scale values to make the shark contour visible and properly proportioned.

3. **Performance optimization**: Redrawing the entire surface for every slider change was initially slow. I optimized this by removing previous plots before drawing new ones rather than recreating the entire figure.

4. **Coordinate system consistency**: Maintaining consistency between mathematical formulas and the visualization library's coordinate system required attention to detail, especially with rotations.

5. **Edge cases**: Handling special cases like empty contours or extreme slider values required adding validation and error handling.

### Q7: How does your implementation handle the mathematical transformations in the right order?
**A:** The order of transformations is crucial in 3D graphics. In my implementation:

1. First, I generate the basic surface points using the mathematical formula
2. Then apply rotations in the specific order: X, then Y, then Z
   - This order matters because rotations are not commutative
3. Finally, apply translations after all rotations
   - This ensures translations occur in the rotated coordinate system

The same transformation sequence is applied to both the surface and the shark contour to maintain their relative positions.

### Q8: What do the parameters R_BASE and R_MOD represent in your surface formula?
**A:** In my implementation:
- **R_BASE** (set to 1.0) is the base radius of the underlying sphere before deformation. It controls the overall size of the surface.
- **R_MOD** (set to 0.3) is the modifier radius that controls the magnitude of the deformation in the x-direction. A larger R_MOD creates a more pronounced asymmetrical bulge that varies with latitude.

Together, these parameters control the shape characteristics of the Moon surface.

### Q9: How would you extend this implementation to texture the surface or add more complex features?
**A:** To extend the implementation, I could:

1. **Add texturing** by:
   - Defining a texture coordinate mapping function
   - Loading an image texture
   - Using plt.pcolormesh or surface plotting with alpha and colormap options

2. **Add lighting** by:
   - Calculating surface normals at each point
   - Implementing a lighting model (e.g., Phong)
   - Modulating colors based on the angle between normals and light direction

3. **Add more complex features** like:
   - Multiple contour mappings
   - Dynamic deformation based on time
   - Additional surface features like craters or mountains using displacement mapping
   - Implementing mesh refinement in areas of high curvature
