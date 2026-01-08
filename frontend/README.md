# SmartCity Computer Vision System - Configuration UI

This is a web-based user interface for configuring the SmartCity Computer Vision system. It allows users to visually define detection zones, adjust parameters, and then export the configuration as a YAML file for use with the main processing pipeline.

## Features

-   **Two Configuration Modes**:
    -   **Simple Mode**: A streamlined view showing only the most essential parameters. Perfect for quick setup and high-level adjustments.
    -   **Detailed Mode**: Exposes all available system parameters for advanced users who need fine-grained control.
-   **Visual Configuration**: Interactively draw and edit polygonal areas on an input image.
-   **Live Parameter Editing**: Modify all system parameters through a structured form.
-   **YAML Import/Export**: Load an existing configuration from a YAML file or download the current settings.
-   **Video Processing Simulation**: Upload a video to get a preview of how the configured system would process it.
-   **State Management**: Session is automatically saved to local storage, so you can resume where you left off.
-   **Rich Editing Tools**: Undo/redo, zoom, pan, snap-to-vertex, and an eraser for fine-tuned control.
-   **Keyboard Shortcuts**: Speed up your workflow with comprehensive keyboard shortcuts.

## Prerequisites

For the backend server (`mainapi.py`) to function correctly, the following are required:

-   **Python 3.8+** with dependencies like FastAPI, Uvicorn, etc.
-   **FFmpeg**: The server requires `ffmpeg` to be installed on the system and accessible from the command line. This is crucial for encoding processed videos into a format that is compatible with web browsers.

## How to Use

### 1. Getting Started

-   **Launch the Application**: Open `index.html` in your web browser.
-   **Load an Image**: The primary step is to load a reference image or a frame from your video source. Click the "upload image" area or drag and drop an image file.
-   **Choose a Configuration Mode**: Use the "Simple" / "Detailed" toggle in the configuration panel to select your preferred level of detail. Your choice will be saved for your next visit.
-   **Load a Configuration (Optional)**: If you have an existing `config.yaml`, you can load it using the "Upload YAML" button in the top right. This will populate all fields and draw any defined areas on the image (if one is loaded).

### 2. The Interface

The screen is split into two main panels:
-   **Left Panel**: The canvas area where you interact with the image and polygons.
-   **Right Panel**: The configuration forms, video processing controls, and system logs.

### 3. Editing Areas (Polygons)

-   **Drawing a New Area**:
    1.  Click the `+ Add Area` button.
    2.  If prompted, enter a name for the new area.
    3.  Click on the image to place vertices.
    4.  To complete the polygon, click on the first vertex (it will be highlighted).
    5.  Press `Escape` to cancel drawing at any time.

-   **Selecting and Modifying an Area**:
    -   Click on a polygon to select it. Its handles (vertices) will become visible and draggable.
    -   Click and drag the handles to move vertices. The vertices will snap to other vertices for easy alignment.
    -   To add a new vertex to an existing polygon, hold `Alt` and click on an edge of the selected polygon.

-   **Deleting an Area**:
    1.  Select a polygon by clicking on it.
    2.  Click the `Delete Selected` button or press the `Delete`/`Backspace` key.

-   **Using the Eraser Tool**:
    1.  Select a polygon.
    2.  Click the `Eraser` button or press `E` to activate the eraser tool.
    3.  Click on any vertex of the selected polygon to remove it. A polygon must maintain at least 3 vertices.
    4.  Press `E` or `Escape` to deactivate the eraser.

-   **Changing Area Order**:
    -   When multiple "Direction Classifier" areas overlap, their order matters.
    -   Select an area and use the `Bring to Front` and `Send to Back` buttons to change its stacking order.

### 4. Changing the Image

-   After loading an image, you can replace it at any time by clicking the `Change Image` button in the canvas toolbar.
-   **Note**: Changing the image will reset all drawn polygons and the undo/redo history.

### 5. Keyboard Shortcuts

| Action                      | Shortcut (PC)            | Shortcut (Mac)          | Notes                                     |
| --------------------------- | ------------------------ | ----------------------- | ----------------------------------------- |
| Undo                        | `Ctrl + Z`               | `Cmd + Z`               | Reverts the last polygon change.          |
| Redo                        | `Ctrl + Y`               | `Cmd + Y`               | Re-applies the last undone change.        |
| **Canvas**                  |                          |                         |                                           |
| Pan                         | Middle Mouse Drag        | Middle Mouse Drag       |                                           |
| Zoom In                     | `Ctrl + =` or `Ctrl + +` | `Cmd + =` or `Cmd + +`  | Zooms in from cursor position.            |
| Zoom Out                    | `Ctrl + -`               | `Cmd + -`               | Zooms out from cursor position.           |
| Reset Zoom                  | `Ctrl + 0`               | `Cmd + 0`               | Resets zoom to 100%.                      |
| **Tools & Editing**         |                          |                         |                                           |
| Activate Eraser             | `E`                      | `E`                     | Only when an area is selected.            |
| Delete Selected Area        | `Delete` / `Backspace`   | `Delete` / `Backspace`  | Deletes the entire selected polygon.      |
| Deselect / Cancel           | `Escape`                 | `Escape`                | Deselects area, cancels drawing/erasing.  |

### 6. Video Processing

1.  Configure your settings and define your areas.
2.  In the "Video Processing" section, click to select a video file (under 100MB).
3.  Click the `Render Video` button.
4.  The application will upload the video and your current configuration to a server for processing.
5.  You can monitor the progress. Once complete, a preview will be shown, and you can download the resulting video.

### 7. Saving Your Work

-   **Session Saving**: Your configuration (including drawn areas and selected mode) is automatically saved in your browser's local storage. If you close the tab and reopen it, your work should be restored.
-   **Clear Session**: To start fresh, click `Clear Session`. This will load the default configuration and cannot be undone.
-   **Downloading YAML**: When you are satisfied with your configuration, click `Download YAML` to save it as a file. This file can then be used by the main SmartCity processing application.