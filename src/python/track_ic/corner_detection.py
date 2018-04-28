import cv2
import numpy as np
from scipy import ndimage as ndi
import skimage
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

def corner_detect(input_image):
    def build_filters():
        filters = []
        ksize = 3
        for theta in np.arange(0, np.pi, np.pi / 16):
            for sigma in (1, 3):
                for freq in (0.05, 0.25):
                    kern = cv2.getGaborKernel((ksize, ksize), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                    kern /= 1.5*kern.sum()  
                    filters.append(kern)
        return filters
    
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)


    def process(img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum
    
    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats
    
    def match(feats, ref_feats):
        min_error = np.inf
        min_i = None
        for i in range(ref_feats.shape[0]):
            error = np.sum((feats - ref_feats[i, :])**2)
            if error < min_error:
                min_error = error
                min_i = i
        return min_i

    def power(image, kernel):
    # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    
    img = input_image

    if img is None:
        print('Failed to load image file!')
        sys.exit(1)

    filters = build_filters()
    power(img, filters)
    res1 = process(img, filters)
    # prepare reference features
    ref_feats = np.zeros((3, len(filters), 2), dtype=np.double)
    ref_feats[0, :, :] = compute_feats(input_image, filters)
    match(res1, ref_feats)
    cv2.imshow('result', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return