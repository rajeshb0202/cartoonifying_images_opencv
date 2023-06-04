import cv2
import numpy as np

#read image
image= cv2.imread('./data/boy.jpg')
grayImage= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



###pencil sketch the image: blur --> Laplacian --> reversing --> eroding --> threshImage
#gaussianImage = cv2.Gaussialur(grayImage, (9,9), 0)
blurImage= cv2.blur(grayImage, (3,3))
edgeImage  = cv2.Laplacian(blurImage, -1)

#normalise the image
normalizeImage = cv2.normalize(edgeImage, None, 0, 255, cv2.NORM_MINMAX)
#reverse the image
revImage = 255 - normalizeImage
#dilating the image
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
#imageErode = cv2.erode(revImage, kernel)
_, threshImage = cv2.threshold(revImage, 240, 255, cv2.THRESH_BINARY)


#combineImages = np.hstack((imageErode, threshImage))
#cv2.imshow("thresholdImage", combineImages)

###applying water colour effect
#blur the color image
blurImage= cv2.blur(image, (9,9))
#combining colour image and sketch image
mergedSketch = cv2.merge((threshImage, threshImage, threshImage))
waterEffectImage = cv2.bitwise_and(blurImage, mergedSketch)


#stacking image for display
stackedImages = np.hstack((mergedSketch, waterEffectImage))
cv2.imshow("cartoonify", stackedImages)

k= cv2.waitKey(-1) & 0xFF
if k == 27:
        cv2.destroyAllWindows()
        
        
        
        
        

        
def waterColourEffect(originalImage, pencilSketch):
        ###applying water colour effect
        #blur the color image
        blurImage= cv2.blur(originalImage, (9,9))
        #combining colour image and sketch image
        mergedSketch = cv2.merge((pencilSketch, pencilSketch, pencilSketch))
        waterEffectImage = cv2.bitwise_and(blurImage, mergedSketch)
        
        return waterEffectImage



cv2.namedWindow("waterPaintingWindow")

#waterEffectImage = waterColourEffect(image,pencilSketchImage)



#cv2.imshow("water_painting", waterEffectImage)