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

---

## Screenshots
<p align="center">
    <img width="500" alt="screenshot (1)" src="https://github.com/user-attachments/assets/24b3ce43-770d-47b3-9985-30dbc469985b" />
    <img width="500" alt="screenshot (2)" src="https://github.com/user-attachments/assets/0ff34430-200b-4737-9bc1-6cb8c8b792b0" />
    <img width="500" alt="screenshot (3)" src="https://github.com/user-attachments/assets/6b0b03b9-4de1-4164-b456-511e582464c6" />

</p>
