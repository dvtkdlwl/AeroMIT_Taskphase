# Volleyball Tracker

Mini project that tracks a volleyball during a match, built using classical OpenCV techniques — (through a lot of trial and error)!

---

## Pipeline

Each frame goes through a basic vision pipeline:

- Resize and preprocess the frame for efficiency  
- Convert from BGR to HSV color space  
- Apply color masking to isolate the volleyball  
- Clean up the mask using morphological operations  
- Detect contours and filter them by size/shape  
- Locate the ball and draw its position on the frame  

---

## Intuition

Using a deep learning model would have yielded much better results, but the goal here was to rely entirely on traditional image-processing techniques.

This approach therefore favors simplicity and interpretability over complexity, and focuses on understanding *why* each step works rather than just getting the best possible accuracy.
