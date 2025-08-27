"""
Visualization utility functions for multi-agent simulation analysis.

This module provides utilities for capturing and saving visual output from
simulation renderers, supporting post-simulation analysis and documentation.
"""

import pygame

def get_snapshot(snapshot, name=''):
    """
    Save a pygame surface as an image file for documentation or analysis.

    This utility function captures a pygame surface (typically from a renderer)
    and saves it as an image file. It's useful for creating documentation,
    generating datasets, or performing post-simulation visual analysis.

    Parameters
    ----------
    snapshot : pygame.Surface
        Pygame surface containing the image to save.
    name : str, optional
        Output filename with extension. If empty, uses default naming.

    Supported Formats
    -----------------
    The function supports various image formats based on file extension:
    
    - **.png**: Portable Network Graphics (recommended for quality)
    - **.jpg/.jpeg**: JPEG format (smaller file size)
    - **.bmp**: Bitmap format (uncompressed)
    - **.tga**: Targa format

    
    Applications
    ------------
    - **Documentation**: Creating figures for papers and reports
    - **Dataset Generation**: Building visual datasets for machine learning
    - **Progress Tracking**: Capturing simulation progress over time
    - **Comparative Analysis**: Visualizing results from different algorithms
    - **Debugging**: Saving problematic states for later analysis
    - **Video Creation**: Individual frames for animation generation

    Notes
    -----
    - Requires pygame to be initialized before use
    - File extension determines output format
    - Creates parent directories if they don't exist (depending on pygame version)
    - Compatible with all pygame surface objects
    - Thread-safe for single-threaded pygame applications
    """
    pygame.image.save(snapshot, name)

