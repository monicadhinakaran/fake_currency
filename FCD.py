import cv2
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from skimage.metrics import structural_similarity as ssim


def upload_and_read_image():
    root = Tk()
    root.withdraw()
    root.update()

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

    if file_path:
        if is_100_rs_note(file_path):
            print("This is likely a ₹100 note.")
            crop_currency(file_path)
        else:
            print("This is not a ₹100 note.")
            sys.exit(0)  # Exit the program if it's not a ₹100 note

        if is_currency_folded(file_path):
            print("The currency note is folded.")
        else:
            print("The currency note is not folded.")

        image = cv2.imread(file_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.imshow(image_rgb)
            plt.title('Uploaded Image')
            plt.axis('off')
            plt.show()

            return file_path
        else:
            print("Error: Could not read the image file.")
    else:
        print("No file selected.")
    return None


def is_100_rs_note(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to HSV (Hue, Saturation, Value) color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of lavender and purple colors in HSV
    lower_lavender = np.array([120, 30, 120])  # Adjust these values based on your specific image
    upper_lavender = np.array([160, 80, 180])

    lower_purple = np.array([130, 30, 100])
    upper_purple = np.array([170, 80, 160])

    # Threshold the HSV image to get only lavender and purple colors
    mask_lavender = cv2.inRange(hsv_image, lower_lavender, upper_lavender)
    mask_purple = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # Combine masks to find lavender or purple pixels
    combined_mask = cv2.bitwise_or(mask_lavender, mask_purple)

    # Count the number of non-zero pixels (lavender or purple)
    count = np.count_nonzero(combined_mask)

    # Determine a threshold count based on your image analysis
    threshold_count = 1000  # Example threshold; adjust based on your image

    # Decide if it's a ₹100 note based on the color detection
    if count >= threshold_count:
        return True
    else:
        return False


def is_currency_folded(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 66, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        print("No contours found!")
        return False

    largest_contour = contours[0]
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        print("Currency note is likely not folded.")
        return False
    else:
        print("Currency note is likely folded.")
        return True


def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def calculate_ssim(edges1, edges2):
    score, _ = ssim(edges1, edges2, full=True, data_range=edges2.max() - edges2.min())
    return score * 100


def intaglio(image_path, template_path, output_dir):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)

    x, y, w, h = 100, 100, 200, 200
    roi = edges[y:y + h, x:x + w]

    image_with_roi = image.copy()
    cv2.rectangle(image_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_path = os.path.join(output_dir, 'intaglio.jpg')
    cv2.imwrite(output_path, image_with_roi)

    template = cv2.imread(template_path, 0)

    template_blurred = cv2.GaussianBlur(template, (5, 5), 0)
    template_edges = cv2.Canny(template_blurred, 50, 150)

    roi_resized = cv2.resize(roi, (template_edges.shape[1], template_edges.shape[0]))

    similarity = calculate_ssim(roi_resized, template_edges)

    return similarity


def security(image_path, template_path, output_dir):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    if image is None or template is None:
        raise ValueError("Error: Could not open or find the image.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
    blurred_template = cv2.GaussianBlur(gray_template, (5, 5), 1.4)

    sobelx_image = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely_image = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined_image = cv2.magnitude(sobelx_image, sobely_image)

    sobelx_template = cv2.Sobel(blurred_template, cv2.CV_64F, 1, 0, ksize=3)
    sobely_template = cv2.Sobel(blurred_template, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined_template = cv2.magnitude(sobelx_template, sobely_template)

    _, thresholded_image = cv2.threshold(sobel_combined_image, 50, 255, cv2.THRESH_BINARY)
    _, thresholded_template = cv2.threshold(sobel_combined_template, 50, 255, cv2.THRESH_BINARY)

    thresholded_image_resized = cv2.resize(thresholded_image, (thresholded_template.shape[1], thresholded_template.shape[0]))

    similarity = calculate_ssim(thresholded_image_resized, thresholded_template)

    output_path = os.path.join(output_dir, 'security.jpg')
    cv2.imwrite(output_path, thresholded_image_resized)

    return similarity


def see_through(image_path, template_path, output_dir):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path, 0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    log_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
    log_edges = np.uint8(np.absolute(log_edges))

    x, y, w, h = 100, 100, 50, 50
    roi = log_edges[y:y + h, x:x + w]

    image_with_roi = image.copy()
    cv2.rectangle(image_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_path = os.path.join(output_dir, 'seethrough_register.jpg')
    cv2.imwrite(output_path, image_with_roi)

    template_blurred = cv2.GaussianBlur(template, (5, 5), 0)
    template_log_edges = cv2.Laplacian(template_blurred, cv2.CV_64F)
    template_log_edges = np.uint8(np.absolute(template_log_edges))

    roi_resized = cv2.resize(roi, (template_log_edges.shape[1], template_log_edges.shape[0]))

    similarity = calculate_ssim(roi_resized, template_log_edges)

    return similarity


def process_images_from_folder(folder_path, template_path, output_dir):
    if not os.path.exists(folder_path):
        print(f"Folder path {folder_path} does not exist.")
        return

    if not os.path.exists(template_path):
        print(f"Template path {template_path} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    intaglio_accuracies = []
    security_accuracies = []
    see_through_accuracies = []

    excel_file = os.path.join(output_dir, r'C:\Users\monic\Downloads\Project\Output\FAKE_CURRENCY.xlsx')

    # Enumerate through files in folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)

            # Process each image
            intaglio_accuracy = intaglio(image_path, template_path, output_dir)
            intaglio_accuracies.append(intaglio_accuracy)

            security_accuracy = security(image_path, template_path, output_dir)
            security_accuracies.append(security_accuracy)

            see_through_accuracy = see_through(image_path, template_path, output_dir)
            see_through_accuracies.append(see_through_accuracy)

    avg_intaglio_accuracy = max(intaglio_accuracies) if intaglio_accuracies else 0
    avg_security_accuracy = max(security_accuracies) if security_accuracies else 0
    avg_see_through_accuracy = max(see_through_accuracies) if see_through_accuracies else 0

    print(f"Average intaglio accuracy: {avg_intaglio_accuracy:.2f}%")
    print(f"Average security thread accuracy: {avg_security_accuracy:.2f}%")
    print(f"Average see-through register accuracy: {avg_see_through_accuracy:.2f}%")

    overall_accuracy = (avg_intaglio_accuracy + avg_security_accuracy + avg_see_through_accuracy) / 3

    if overall_accuracy >= 75:
        print(f"The currency is absolutely real with an overall accuracy of {overall_accuracy:.2f}%.")
    elif overall_accuracy >= 65:
        print(f"The currency is slightly real with an overall accuracy of {overall_accuracy:.2f}%.")
    else:
        print(f"The currency may be fake with an overall accuracy of {overall_accuracy:.2f}%.")

    # Determine if currency is considered real or fake
    is_real = "Yes" if overall_accuracy >= 60 else "No"
    is_fake = "No" if overall_accuracy >= 60 else "Yes"

    # Determine if intaglio accuracy passes the threshold
    intaglio_status = "Pass" if avg_intaglio_accuracy > 60 else "Fail"

    # Determine if see-through register accuracy passes the threshold
    see_status = "Pass" if avg_see_through_accuracy > 60 else "Fail"

    # Determine if security thread accuracy passes the threshold
    sec_status = "Pass" if avg_security_accuracy > 60 else "Fail"

    # Store the summary data in a DataFrame
    data = {
        'Serial Number': [+1],
        'Filename': [template_path],
        'Intaglio Accuracy': [avg_intaglio_accuracy],
        'Intaglio Status': [intaglio_status],
        'Security Thread Accuracy': [avg_security_accuracy],
        'Security Thread Status': [sec_status],
        'See-Through Register Accuracy': [avg_see_through_accuracy],
        'See-Through Register Status': [see_status],
        'Accuracy': [overall_accuracy],
        'Is_real': [is_real],
        'Is_fake': [is_fake]
    }
    df_new = pd.DataFrame(data)

    # Check if the Excel file already exists and read it if it does
    if os.path.exists(excel_file):
        df_existing = pd.read_excel(excel_file)
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_existing = df_new

    # Save the DataFrame to an Excel file
    df_existing.to_excel(excel_file, index=False)
    print(f"Accuracy data saved to {excel_file}")


def crop_currency(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate the edges to fill gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image using the bounding rectangle coordinates
    cropped_img = img[y:y + h, x:x + w]

    # Save the cropped image
    cropped_image_path = r"C:\Users\monic\Downloads\Project\Output\crop-img.jpg"
    cv2.imwrite(cropped_image_path, cropped_img)

    # Display the cropped image using matplotlib
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    plt.imshow(cropped_img_rgb)
    plt.title("Cropped Currency")
    plt.axis('off')
    plt.show()

    print(f"Cropped image saved as {cropped_image_path}")


def main():
    template_path = upload_and_read_image()

    folder_path = r'C:\Users\monic\Downloads\Project\Trained Dataset'
    output_dir = r'C:\Users\monic\Downloads\Project\Output'

    if template_path:
        process_images_from_folder(folder_path, template_path, output_dir)


if __name__ == "__main__":
    main()
