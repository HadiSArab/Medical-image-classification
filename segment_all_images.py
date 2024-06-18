
import numpy as np
from PIL import Image
# import required module
from pathlib import Path

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


# get the path/directory
folder_dir = 'D:\Github\OASIS Dataset\input\Mild Dementia'

# iterate over files in
# that directory
images = Path(folder_dir).glob('*.jpg')
for image in images:
    im = np.array(Image.open(image).convert('L'))
    im_otsu = threshold_image(im, find_best_threshold(im))
    name = (str(image).split('\\')[-1])
    print(type(im))
    print(type(im_otsu))
    im_otsu.save(f'D:\Github\OASIS Dataset\prepared_dataset\Mild Dementia\{name}')
    break
