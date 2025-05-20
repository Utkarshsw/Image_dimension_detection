import cv2
import numpy as np
import imutils
from imutils.perspective import order_points


# Load the input image
img = cv2.imread("image_assignment.jpg")
img_resized = imutils.resize(img, width=1000)

# Convert to grayscale and apply edge detection
gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
edges = cv2.Canny(blurred_img, 50, 100)
edges = cv2.dilate(edges, None, iterations=1)
edges = cv2.erode(edges, None, iterations=1)

# Identify contours in the image
contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Use visiting card as scale reference (88mm x 50mm)
scale_factor = None

for contour in contours:
    if cv2.contourArea(contour) < 1000:
        continue

    # Obtain the rotated bounding box
    rect = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(rect)
    box = np.array(box_points, dtype="int")
    box = order_points(box)
    (tl, tr, br, bl) = box

    # Calculate object width and height in pixels
    width_pixels = np.linalg.norm(tr - tl)
    height_pixels = np.linalg.norm(tr - br)

    # Determine scale from visiting card
    if scale_factor is None:
        scale_x = width_pixels / 88.0
        scale_y = height_pixels / 50.0
        scale_factor = (scale_x + scale_y) / 2
        print(f"[INFO] Scale (pixels per mm): {scale_factor:.2f}")
        continue

    # Convert pixel measurements to mm
    width_mm = width_pixels / scale_factor
    height_mm = height_pixels / scale_factor

    # Annotate image with dimensions
    cv2.drawContours(img_resized, [box.astype("int")], -1, (0, 255, 0), 2)
    label = f"{width_mm:.1f}mm x {height_mm:.1f}mm"
    cv2.putText(img_resized, label, (int(tl[0]), int(tl[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the final annotated image
cv2.imshow("Measured Objects", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
