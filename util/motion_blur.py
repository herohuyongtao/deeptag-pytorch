import cv2
import numpy as np

def motion_blur(image, kernel_size, angle):

    M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(kernel_size))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel_size, kernel_size))
 
    motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel)
    blur = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # cv2.normalize(blur, blur, 0, 255, cv.NORM_MINMAX)
 
    return blur

if __name__ == '__main__':

    nb_test = 50

    image = np.zeros( (512, 512, 3), dtype = np.float32)
    image[64:512-64, 64:512-64,:] =1
    image[128:512-128, 128:512-128,:] =0
    
    image = cv2.resize(image, (200,200))
    
    cv2.imshow('image', image)

    for _ in range(nb_test):

        k_size = np.random.choice(np.arange(1, 21, 2))
        ang = np.random.choice(np.arange(-45, 45))
        print(k_size, ang)
 
        res = motion_blur(np.copy(image), k_size, ang)
        cv2.imshow('blur', res)
        c = cv2.waitKey(0)
        if c == 27 or c == ord('q'): break
