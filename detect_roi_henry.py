from typing import Tuple

import numpy as np

from skimage import color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.morphology import dilation, square
from skimage.measure import label
from matplotlib.patches import Rectangle
import scipy.ndimage as ndimage

import SimpleITK as sitk


class RegionOfInterest():
    
    def __init__(self):
        pass
    
    
    @staticmethod
    def _mean_roi_centroid(centroids_x: Tuple[int], centroids_y: Tuple[int]) -> Tuple[int]:
        centroid_x = int(np.asarray(centroids_x).mean())    
        centroid_y = int(np.asarray(centroids_y).mean())
        
        return centroid_x, centroid_y
    
    
    @staticmethod
    def detect_roi_sa(sitk_image: sitk.Image,
                      debug: bool = False) -> Tuple[int]:
        image = sitk.GetArrayFromImage(sitk_image)
        image = np.swapaxes(image, 0, -1)
        
        all_cx = []
        all_cy = []

        ed_slice = image[:, :, 0]
        es_slice = image[:, :, 10]
        
        width = ed_slice.shape[0]
        height = ed_slice.shape[1]
        image_size = (width + height) // 2
            
        diff_image = abs(ed_slice - es_slice)
        edge_image = canny(diff_image, sigma=2.0, low_threshold=0.8, high_threshold=0.98,
                           use_quantiles=True)
        
        lower_range = int(image_size * 0.06)    
        upper_range = int(image_size * 0.08)
        hough_radii = np.arange(lower_range, upper_range, 20)
        hough_res = hough_circle(edge_image, hough_radii)
        
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=3,
                                                   normalize=False)
        
        all_cx.extend(cx)
        all_cy.extend(cy)
        
        if debug:
            import matplotlib.pyplot as plt
            
            plt.imshow(ed_slice, cmap='gray')
            plt.title('Passed End Diastolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            plt.imshow(es_slice, cmap='gray')
            plt.title('Passed End Systolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(diff_image, cmap='magma')
            plt.title('Difference between ED and ES')
            plt.axis('off')
            plt.show()
            plt.close()
             
            plt.imshow(edge_image, cmap='cubehelix')
            plt.title('Detected Edges on Difference Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            mean_cx, mean_cy = RegionOfInterest._mean_roi_centroid(cx, cy)
            
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            image = ((ed_slice - ed_slice.min()) *
                     (1 / (ed_slice.max() - ed_slice.min()) * 255)).astype('uint8')
            image = color.gray2rgb(image)
            for center_y, center_x, radius in zip(cy, cx, radii):
                circy, circx = circle_perimeter(center_y, center_x, radius,
                                                shape=image.shape)
                image[circy, circx] = (220, 60, 40)
            
            ax.imshow(image, cmap=plt.cm.gray)
            ax.scatter([mean_cx], [mean_cy], marker='x')
            plt.title('Detected Hough Circles and Centroid')
            plt.axis('off')
            plt.show()
            plt.close()
            
        mean_cx, mean_cy = RegionOfInterest._mean_roi_centroid(all_cx, all_cy)
        
        return mean_cy, mean_cx

    
    @staticmethod
    def detect_roi_la(sitk_image: sitk.Image,
                      debug: bool = True) -> Tuple[int]:
        image = sitk.GetArrayFromImage(sitk_image)
        image = np.swapaxes(image, 0, -1)
        
        ed_slice = image[:, :, 0]
        es_slice = image[:, :, 10]
        
        diff_image = abs(ed_slice - es_slice)
        edge_image = canny(diff_image, sigma=2.0, low_threshold=0.6, high_threshold=0.96,
                           use_quantiles=True)
        
        # The accuracy corresponds to the bin size of a major axis.
        # The value is chosen in order to get a single high accumulator.
        # The threshold eliminates low accumulators
        # Min_size: Minimal major axis length
        # Max_size: Maximal minor axis length
        # result = hough_ellipse(edge_image, accuracy=150, threshold=500)
        # result.sort(order='accumulator')
        
        # best = list(result[-1])
        # yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        # orientation = best[5]
        
        # cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        
        width = ed_slice.shape[1]
        height = ed_slice.shape[0]
        image_size = (width + height) // 2
        
        lower_range = int(image_size * 0.1)    
        upper_range = int(image_size * 0.3)
        hough_radii = np.arange(lower_range, upper_range, 3)
        hough_res = hough_circle(edge_image, hough_radii)
        
        
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                    total_num_peaks=10,
                                                    normalize=False) 
        
        mean_cx, mean_cy = RegionOfInterest._mean_roi_centroid(cx, cy)
        
        if debug:
            import matplotlib.pyplot as plt
            
            plt.imshow(ed_slice, cmap='bone')
            plt.title('Passed End Diastolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            plt.imshow(es_slice, cmap='bone')
            plt.title('Passed End Systolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(diff_image, cmap='magma')
            plt.title('Difference between ED and ES')
            plt.axis('off')
            plt.show()
            plt.close()
             
            plt.imshow(edge_image, cmap='cubehelix')
            plt.title('Detected Edges on Difference Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            mean_cx, mean_cy = RegionOfInterest._mean_roi_centroid(cx, cy)
            
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            ed_slice[ed_slice > 350] = 350
            image = ((ed_slice - ed_slice.min()) *
                     (1 / (ed_slice.max() - ed_slice.min()) * 255)).astype('uint8')
            image = color.gray2rgb(image)
            for center_y, center_x, radius in zip(cy, cx, radii):
                circy, circx = circle_perimeter(center_y, center_x, radius,
                                                shape=image.shape)
                image[circy, circx] = (220, 60, 40)
            
            ax.imshow(image, cmap=plt.cm.gray)
            ax.scatter([mean_cx], [mean_cy], marker='x')
            plt.title('Detected Hough Circles and Centroid')
            plt.axis('off')
            plt.show()
            plt.close()
            
            # image[cy, cx] = (0, 0, 255)
            # edges = color.gray2rgb(img_as_ubyte(edge_image))
            # edges[cy, cx] = (250, 0, 0)
            
            # fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
            #                                 sharex=True, sharey=True)
            # ax1.set_title('Original picture')
            # ax1.imshow(image, cmap = 'bone')
            # ax2.set_title('Edge (white) and result (red)')
            # ax2.imshow(edges)
            # plt.show()
         
        
        return mean_cx, mean_cy, edge_image
    
    
    @staticmethod
    def detect_roi_dilate_n_crop(image, debug):#  sitk_image: sitk.Image,
                     # debug: bool = True) -> Tuple[int]:
        # image = sitk.GetArrayFromImage(sitk_image)
        
        ed_slice = image[:, :, 0]
        try:
            es_slice = image[:, :, 10]
        except:
            es_slice = image[:, :, -1]
        else:
            es_slice = image[:, :, 10]
        
        diff_image = abs(ed_slice - es_slice)
        edge_image = canny(diff_image, sigma=2.0, low_threshold=0.6, high_threshold=0.96,
                           use_quantiles=True)
        
        edge_image_dilated = dilation(edge_image, square(5))
        
        labels = label(edge_image_dilated)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        
        temp_top = largestCC.shape[0]
        temp_bottom = 0
        temp_left = largestCC.shape[1]
        temp_right = 0
        for (row_index, row) in enumerate(largestCC):
            for (col_index, col) in enumerate(row):
                if col > 0:
                    # Get the top-most coordinate
                    if row_index < temp_top:
                        temp_top = row_index
                    # Get the bottom-most coordinate
                    if row_index > temp_bottom:
                        temp_bottom = row_index
                    # Get the left-most coordinate
                    if col_index < temp_left:
                        temp_left = col_index
                    # Get the right-most coordinate
                    if col_index > temp_right:
                        temp_right = col_index
        box = [temp_left, temp_right, temp_top, temp_bottom]
        rect = Rectangle((box[0],box[2]),(box[1]-box[0]),(box[3]-box[2]),linewidth=1,edgecolor='r',facecolor='none')
        
        cropped_image = image[box[2]:box[3], box[0]:box[1]]
        
        if debug:
            import matplotlib.pyplot as plt
            
            plt.imshow(ed_slice, cmap='bone')
            plt.title('Passed End Diastolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            plt.imshow(es_slice, cmap='bone')
            plt.title('Passed End Systolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(diff_image, cmap='magma')
            plt.title('Difference between ED and ES')
            plt.axis('off')
            plt.show()
            plt.close()
             
            plt.imshow(edge_image, cmap='cubehelix')
            plt.title('Detected Edges on Difference Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(edge_image_dilated, cmap='cubehelix')
            plt.title('Dilated Edges Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(largestCC, cmap='cubehelix')
            ax = plt.gca()
            ax.add_patch(rect)
            plt.title('Largest Connected Component')
            plt.axis('off')
            plt.show()
            plt.close()
        
        return box
    
    @staticmethod
    def detect_roi_dilate_n_crop_2ch(sitk_image: sitk.Image,
                      debug: bool = True) -> Tuple[int]:
        image = sitk.GetArrayFromImage(sitk_image)
        image = np.swapaxes(image, 0, -1)
        
        ed_slice = image[:, :, 0]
        try:
            es_slice = image[:, :, 10]
        except:
            es_slice = image[:, :, -1]
        else:
            es_slice = image[:, :, 10]
        
        diff_image = abs(ed_slice - es_slice)
        edge_image = canny(diff_image, sigma=2.0, low_threshold=0.6, high_threshold=0.96,
                           use_quantiles=True)
        
        width = ed_slice.shape[0]
        height = ed_slice.shape[1]
        image_size = (width + height) // 2
        
        lower_range = int(image_size * 0.06)    
        upper_range = int(image_size * 0.08)
        hough_radii = np.arange(lower_range, upper_range, 20)
        hough_res = hough_circle(edge_image, hough_radii)

        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=3,
                                                   normalize=False)
        
        img = ((ed_slice - ed_slice.min()) *
                 (1 / (ed_slice.max() - ed_slice.min()) * 255)).astype('uint8')
        
        circle_mask = np.zeros(img.shape)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=img.shape)
            circle_mask[circy, circx] = 1
            circle_mask = ndimage.binary_fill_holes(circle_mask)
        
        circle_mask = ~circle_mask
        circle_mask = np.array(circle_mask, 'int8')
        
        masked_circle_edge_image = edge_image * circle_mask
        
        edge_image_dilated = dilation(masked_circle_edge_image, square(5))
        
        labels = label(edge_image_dilated)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        
        temp_top = largestCC.shape[0]
        temp_bottom = 0
        temp_left = largestCC.shape[1]
        temp_right = 0
        for (row_index, row) in enumerate(largestCC):
            for (col_index, col) in enumerate(row):
                if col > 0:
                    # Get the top-most coordinate
                    if row_index < temp_top:
                        temp_top = row_index
                    # Get the bottom-most coordinate
                    if row_index > temp_bottom:
                        temp_bottom = row_index
                    # Get the left-most coordinate
                    if col_index < temp_left:
                        temp_left = col_index
                    # Get the right-most coordinate
                    if col_index > temp_right:
                        temp_right = col_index
        box = [temp_left, temp_right, temp_top, temp_bottom]
        rect = Rectangle((box[0],box[2]),(box[1]-box[0]),(box[3]-box[2]),linewidth=1,edgecolor='r',facecolor='none')
        
        cropped_image = image[box[2]:box[3], box[0]:box[1]]
        
        if debug:
            import matplotlib.pyplot as plt
            
            plt.imshow(ed_slice, cmap='bone')
            plt.title('Passed End Diastolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            plt.imshow(es_slice, cmap='bone')
            plt.title('Passed End Systolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(diff_image, cmap='magma')
            plt.title('Difference between ED and ES')
            plt.axis('off')
            plt.show()
            plt.close()
             
            plt.imshow(edge_image, cmap='cubehelix')
            plt.title('Detected Edges on Difference Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(masked_circle_edge_image, cmap='cubehelix')
            plt.title('Aorta Removal Edges Images')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(edge_image_dilated, cmap='cubehelix')
            plt.title('Dilated Edges Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(largestCC, cmap='cubehelix')
            ax = plt.gca()
            ax.add_patch(rect)
            plt.title('Largest Connected Component')
            plt.axis('off')
            plt.show()
            plt.close()
        
        return box

    
