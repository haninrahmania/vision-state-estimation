# Vision State Estimation (Initial Version)
This project explores real-time visual perception for robotics using webcam input.

### Features
- Real-time webcam capture with OpenCV
- HSV-based color thresholding for object segmentation
- Largest-blob detection and centroid extraction
- Kalman filter state estimation:
- raw measurement
- filtered estimate
- motion prediction
- Interactive color calibration via mouse click
  - Click on an object in the tracking window
  - HSV range is automatically generated
  - Eliminates manual slider tuning

### Visualization
- Red → raw measurement
- Green → filtered estimate
- Blue → predicted position

## Goal
This project represents a perception module intended to complement navigation and planning systems.