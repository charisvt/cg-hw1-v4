# Triangle Filling with Flat Shading and Texture Mapping

This project implements triangle filling algorithms using scanline rendering. The implementation includes flat shading and texture mapping techniques.

## Features

- **Flat Shading**: Fill triangles with a solid color calculated as the average color of the triangle's vertices.
- **Texture Mapping**: Apply texture coordinates (UV) to triangles and sample colors from a texture image.


## How to Run

1. Flat shading:
```
python demo_f.py
```

2. Texture shading:
```
python demo_t.py
```

## File Structure

- `functions.py`: Contains the core implementation of shading algorithms:
  - `f_shading()`: Implements flat shading for triangles
  - `t_shading()`: Implements texture mapping for triangles
  - `render_img()`: Main rendering function
  
- `demo_f.py`: Demo for flat shading
- `demo_t.py`: Demo for texture mapping
- `hw1.npy`: Input data containing vertices, colors, and triangle indices
- `texImg.jpg`: Texture image for mapping

## Requirements

- NumPy
- OpenCV (cv2)