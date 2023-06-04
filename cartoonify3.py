import cv2
import numpy as np

#read images and convert into float
image = cv2.imread('./data/boy.jpg', cv2.IMREAD_UNCHANGED)
image = np.float32(image)/255
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#make a new window
cv2.namedWindow("pencil sketch")


#trackbar event function- calls other functions when a trackbar value is changed
def onChange(x):
    kSize = cv2.getTrackbarPos("Kernel Size", "pencil sketch")
    if (kSize%2 == 0):
        kSize +=1

    sketch(kSize)

#function 
def sketch(kSize):
    global grayImage, image
    blurImage = cv2.GaussianBlur(grayImage, (kSize, kSize), 0)
    
    #calling different order gradient function to produce sketch images
    gradSobel = sobel(blurImage)
    gradLap = laplacian(blurImage)   
    stackedGradImages= np.hstack((gradSobel, gradLap))
    cv2.imshow("pencil sketch",stackedGradImages)

    #calling cartoonify function
    cartoonSobel = cartoonify(gradSobel)
    cartoonLap = cartoonify(gradLap)
    stackedCartoonImage = np.hstack((cartoonSobel, cartoonLap))
    cv2.imshow("cartoonised", stackedCartoonImage)


#function to compute Sobel gradients and threshold it
def sobel(blurImage):
    sobelX = cv2.Sobel(blurImage, -1, 1, 0)
    sobelY = cv2.Sobel(blurImage, -1, 0, 1)
    sobelImage = np.sqrt(np.power(sobelX,2) + np.power(sobelY,2))
    normalisedImage = cv2.normalize(sobelImage, None, 0,1, cv2.NORM_MINMAX)
    _, threshSobel = cv2.threshold(normalisedImage, 0.5, 1, cv2.THRESH_BINARY_INV)
    return threshSobel


#function to compute Laplacian gradients and threshold it
def laplacian(blurImage):
    laplacian = cv2.Laplacian(blurImage, cv2.CV_32F)
    normalisedLaplacian = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
    _, threshLap = cv2.threshold(normalisedLaplacian, 0.5,1, cv2.THRESH_BINARY_INV)
    return threshLap


#function to produce a cartoon image
def cartoonify(gradImage):
    global grayImage, image
    blurImage = cv2.GaussianBlur(image, (5,5), 0)
    gradImage = cv2.merge((gradImage, gradImage, gradImage))
    cartoonImage = cv2.bitwise_and(gradImage, blurImage)
    return cartoonImage



#create trackbars
cv2.createTrackbar("Kernel Size", "pencil sketch", 3, 15, onChange)


q = cv2.waitKey(-1) & 0xFF
if q == 27:
    cv2.destroyAllWindows()