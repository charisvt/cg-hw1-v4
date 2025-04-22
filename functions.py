import numpy as np

def vector_interp(p1, p2, V1, V2, coord, dim):
    """
    Linearly interpolates vector V at a point `p` on the line segment from p1 to p2.
    V can be n-dimensional.

    Parameters:
        p1 (tuple): Coordinates (x1, y1) of the first point.
        p2 (tuple): Coordinates (x2, y2) of the second point.
        V1 (array-like): Vector value at p1.
        V2 (array-like): Vector value at p2.
        coord (float): The x or y coordinate of the point `p` to interpolate at.
        dim (int): 1 for x-coordinate, 2 for y-coordinate.
    
    Returns:
        numpy.ndarray: Interpolated vector value at point `p`.
    """
    # Select the dimension
    c1 = p1[dim - 1]
    c2 = p2[dim - 1]

    # Handle division by zero (coordinates equal)
    if c1 == c2:
        return np.array(V1)

    # Interpolation factor
    t = (coord - c1) / (c2 - c1)

    # Linear interpolation
    return np.array(V1) + t * (np.array(V2) - np.array(V1))
    

def vector_mean(v1, v2, v3):
    """
    Calculates the mean of three vectors.
    """
    return (v1 + v2 + v3) / 3


def f_shading(img, vertices, vcolors):
    """
    Performs flat shading on a triangle.
    The entire triangle is colored with the average color of its vertices.
    Uses scanline algorithm to fill the triangle.
    """
    updated_img = img.copy()
    height, width = img.shape[:2]
    triangle = vertices.tolist()
    flat_color = vector_mean(vcolors[0], vcolors[1], vcolors[2])

    ymin = max(int(np.min(vertices[:, 1])), 0)
    ymax = min(int(np.max(vertices[:, 1])) + 1, height)

    for y in range(ymin, ymax):
        intersections = []
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i + 1) % 3]
            y1, y2 = p1[1], p2[1]

            if y1 == y2:
                continue  # Skip horizontal edges

            if (y >= y1 and y < y2) or (y >= y2 and y < y1):
                x = p1[0] + (y - y1) * (p2[0] - p1[0]) / (y2 - y1)
                intersections.append(x)

        if len(intersections) < 2:
            continue

        intersections.sort()
        x_start = max(int(np.ceil(intersections[0])), 0)
        x_end = min(int(np.floor(intersections[1])), width - 1)

        updated_img[y, x_start:x_end + 1] = flat_color

    return updated_img


def t_shading(img, vertices, uv, textImg):
    """
    Performs texture mapping on a triangle.
    Uses scanline algorithm with two phases:
    1. Find intersection points with triangle edges and their UV coordinates
    2. For each scanline, interpolate UV coordinates and sample texture
    """
    img = img.copy()
    M, N, _ = img.shape
    K, L, _ = textImg.shape
    
    v0, v1, v2 = vertices
    uv0, uv1, uv2 = uv

    ymin = max(int(np.floor(min(v0[1], v1[1], v2[1]))), 0)
    ymax = min(int(np.ceil(max(v0[1], v1[1], v2[1]))), M-1)

    # Define edges with their corresponding UV coordinates
    edges = [(v0, v1, uv0, uv1), (v1, v2, uv1, uv2), (v2, v0, uv2, uv0)]

    for y in range(ymin, ymax+1):
        # Phase 1: Find intersection points and their UV coordinates
        intersections = []
        uvs = []
        
        for p1, p2, uv1, uv2 in edges:
            if p1[1] == p2[1]:
                continue  # Skip horizontal edges
                
            if (y >= min(p1[1], p2[1])) and (y <= max(p1[1], p2[1])):
                # Calculate x intersection
                x = vector_interp(p1, p2, p1[0], p2[0], y, 2)
                # Calculate UV coordinates at intersection
                uv_interp = vector_interp(p1, p2, uv1, uv2, y, 2)
                intersections.append(x)
                uvs.append(uv_interp)

        if len(intersections) == 2:
            # Sort intersections and UVs by x coordinate
            if intersections[0] > intersections[1]:
                intersections[0], intersections[1] = intersections[1], intersections[0]
                uvs[0], uvs[1] = uvs[1], uvs[0]

            x_start, x_end = intersections
            uv_start, uv_end = uvs

            # Ensure x coordinates are within image bounds
            x_start = max(int(np.ceil(x_start)), 0)
            x_end = min(int(np.floor(x_end)), N-1)

            # Phase 2: Interpolate UV coordinates for each point on the scanline
            for x in range(x_start, x_end+1):
                # Interpolate UV coordinates based on x position
                uv_interp = vector_interp((x_start, y), (x_end, y), uv_start, uv_end, x, 1)
                u, v = uv_interp
                
                # Convert to texture coordinates
                tx = int(np.floor(u * (L - 1)))
                ty = int(np.floor(v * (K - 1)))
                
                # Sample texture using nearest neighbor
                img[y, x] = textImg[ty, tx]

    return img

def render_img(faces, vertices, vcolors, uvs, depth, shading, textImg):
    """
    Main rendering function that handles both flat and texture shading.
    Process:
    1. Initialize white canvas
    2. Sort faces by depth (back to front)
    3. For each face:
       - Get triangle vertices, colors, and UVs
       - Apply appropriate shading (flat or texture)
    4. Normalize and convert to uint8 format
    """
    # Declare canvas size
    M = 512
    N = 512
    img = np.ones((M, N, 3), dtype=np.float32)

    # Sort faces by average depth (farthest to closest)
    avg_depth = np.mean(depth[faces], axis=1)
    sorted_indices = np.argsort(avg_depth)[::-1]  # Render from back to front

    for idx in sorted_indices:
        face = faces[idx]

        triangle_vertices = vertices[face]
        triangle_colors = vcolors[face]
        triangle_uvs = uvs[face] if uvs is not None else None

        if shading == 'f':
            img = f_shading(img, triangle_vertices, triangle_colors)
        elif shading == 't':
            img = t_shading(img, triangle_vertices, triangle_uvs, textImg)
        else:
            raise ValueError("Invalid shading mode. Use 'f' for flat or 't' for texture.")

    # Clamp values to [0, 1] before scaling to avoid overflows
    img = np.clip(img, 0, 1)

    # Normalize the canvas and convert it to the appropriate format for saving with OpenCV
    img_normalized = (img * 255).astype(np.uint8)

    return img_normalized