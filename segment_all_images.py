import cv2 as cv
import glob

Directories = ["Mild Dementia","Moderate Dementia","Non Demented","Very mild Dementia"]
for Dname in Directories:
   
   # Find all JPG files in the current directory
   images = glob.glob(f"input\{Dname}\*.jpg")
   print("\n\n\n\n\n\n\n\n\n Directory : ",Dname)

   for image in images:
      img = cv.imread(image, cv.IMREAD_GRAYSCALE)
      assert img is not None, "file could not be read, check with os.path.exists()"
      new_width = 196
      new_height = 196
      img = cv.resize(img, (new_width, new_height))
      img = cv.medianBlur(img,5)
      
      th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
      cv.THRESH_BINARY,11,2)
      name = image.split("\\")[-1]
      # print(name)

      cv.imwrite(f"prepared_dataset\{Dname}\{name}",th)