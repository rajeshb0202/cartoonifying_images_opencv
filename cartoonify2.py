import cv2
import numpy as np

kSize= 3
iterations= 1
threshValue= 200

cv2.namedWindow("pencilSketchWindow")


def on_change(x):
        global grayImage, image    
        global kSize, iterations, thresh_value
        kSize = cv2.getTrackbarPos("Kernel Size Bar","pencilSketchWindow")
        iterations = cv2.getTrackbarPos("Erode Iterations Bar","pencilSketchWindow")
        threshValue = cv2.getTrackbarPos("Threshold Value Bar","pencilSketchWindow")
        
        pencilSketchImage= pencil_sketch(kSize, iterations, threshValue)
        
        
        
        blurImage= cv2.blur(image, (10,10))
        #combining colour image and sketch image
        mergedSketch = cv2.merge((pencilSketchImage, pencilSketchImage, pencilSketchImage))
        cartoonImage = cv2.bitwise_and(blurImage, mergedSketch)
        
        combinedImages= np.hstack((image, mergedSketch, cartoonImage))
        
        cv2.imshow("pencilSketchWindow", combinedImages)
        

#function for pencilsketch
def pencil_sketch(kSize, iterations, threshValue):
        #global kSize, iterations, threshValue
        global grayImage
        
        ###pencil sketch the image: blur --> Laplacian --> normalising --> reversing --> eroding --> threshImage
        if (kSize%2 == 0):
                kSize += 1
        gaussianImage = cv2.GaussianBlur(grayImage, (kSize,kSize), 0)
       
        edgeImage  = cv2.Laplacian(gaussianImage, -1)
        

        normalizeImage = cv2.normalize(edgeImage, None, 0, 255, cv2.NORM_MINMAX)
       
        revImage = 255 - normalizeImage
       
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        imageErode = cv2.erode(revImage, kernel,iterations = iterations)
        
        _, threshImage = cv2.threshold(imageErode, threshValue, 255, cv2.THRESH_BINARY)
        
              
        return threshImage
        
        
      


        

#read image
image= cv2.imread('./data/trump.jpg')
grayImage= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#creating trackbars
cv2.createTrackbar("Kernel Size Bar","pencilSketchWindow", 3, 21, on_change)
cv2.createTrackbar("Erode Iterations Bar","pencilSketchWindow", 1, 10, on_change)
cv2.createTrackbar("Threshold Value Bar","pencilSketchWindow", 200, 255, on_change)





k= cv2.waitKey(-1) & 0xFF
if k == 27:
        cv2.destroyAllWindows()

