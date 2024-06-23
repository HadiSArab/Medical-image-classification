# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
 

# img = cv.imread('D:\Computer Vision\Project\\first\img\org.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# img = cv.medianBlur(img,5)
 
# ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#  cv.THRESH_BINARY,11,2)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#  cv.THRESH_BINARY,11,2)
 
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
 
# for i in range(4):
#  plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#  plt.title(titles[i])
#  plt.xticks([]),plt.yticks([])
# plt.show()



###################################################################


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

def threshold_image (im, th):
   thresholded_im = np.zeros(im.shape)
   thresholded_im[im >= th] = 1
   return thresholded_im

def compute_otsu_criteria(im, th):
   thresholded_im = threshold_image(im, th)
   nb_pixels = im.size
   nb_pixels1 = np.count_nonzero(thresholded_im)
   weight1 = nb_pixels1 / nb_pixels
   weight0 = 1-weight1
   if weight1 == 0 or weight0 == 0:
      return np.inf
   
   val_pixels1 = im[thresholded_im == 1]
   val_pixels0 = im[thresholded_im == 0]
   var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
   var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
   return weight0 * var0 + weight1 * var1


def find_best_threshold(im):
   threshold_range = range(np.max(im)+1)
   criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
   best_threshold = threshold_range[np.argmin(criterias)]
   return best_threshold


path_image = 'Side-By-Side-Of-Brain-MRI-Scan-Results.webp'
im = np.array(Image.open(path_image).convert('L'))
im_otsu = threshold_image(im, find_best_threshold(im))



image = mpimg.imread(path_image)
# plt.imshow(image)
# plt.show()

plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
plt.title('original image')
plt.imshow(im)
plt.subplot(1,3,2)
plt.title('original gray image')
plt.imshow(im,cmap='gray')
plt.subplot(1,3,3)
plt.title('otsu image')
plt.imshow(im_otsu,cmap='gray')
plt.tight_layout()
plt.show()