from PIL import Image
import os

# Path to the folder containing images
folder_path = "path_to_your_folder"

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):  # Check if the file is an image
        # Open the image file
        image_path = os.path.join(folder_path, file_name)
        img = Image.open(image_path)
        # grayscaling
        img = img.convert('L')

        # Resize the image to 196x196 pixels
        img_resized = img.resize((196, 196))

        # Save the resized image
        img_resized.save(os.path.join(folder_path, f"resized_{file_name}"))

print("Image resizing complete.")
