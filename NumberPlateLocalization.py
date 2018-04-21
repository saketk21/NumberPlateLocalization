import os
import sys
import numpy as np
import cv2

# Prerequisite - Defining the File Path for the Dataset
# and Creating a Folder to hold Output Images

DATASET_PATH = os.path.join(os.getcwd(), 'Dataset')
IMAGE_LIST = os.listdir(DATASET_PATH)
if(not os.path.isdir('FinalOutput')):
    os.mkdir('FinalOutput')
OUTPUT_PATH = os.path.join(os.getcwd(), 'FinalOutput')

# Reading the Image
def readImage(imagePath):
    image = cv2.imread(imagePath)
    return image

# Converting Image to Grayscale
def convertImageToGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise Removal from the image by Bilateral Filtering
def noiseRemoval(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

# Histogram Equalization of the Image
def histogramEqualization(image):
    return cv2.equalizeHist(image)

# Morphological Opening of the Image using Provided Structuring Element
def morphologicalOpening(image, structElem):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, structElem, iterations=15)

# Subtract Morphologically Opened Image from Histogram Equalized Image
def subtractOpenFromHistEq(histEqImage, morphImage):
    return cv2.subtract(histEqImage, morphImage)

# Thresholding the Image
def thresholdingImage(image):
    ret, threshImage = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return threshImage

# Applying Canny Edge Detection
def edgeDetection(image, threshold1, threshold2):
    cannyImage = cv2.Canny(image, threshold1, threshold2)
    cannyImage = cv2.convertScaleAbs(cannyImage)
    return cannyImage

# Dilation for Edge Strengthening
def imageDilation(image, structElem):
    return cv2.dilate(image, structElem, iterations=1)

# Finding Contours of the Edge Dilated Image, which will find edges
def findContours(image):
    newImage, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For this problem, number plate should have contours with a small area as compared to other contours.
    # Hence, we sort the contours on the basis of contour area and take the least 10 contours
    return sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Approximate the Polygon of the Contours using Ramer-Douglas-Peucker Algorithm
# Break on obtaining a quadrilateral
def approximateContours(contours):
    approximatedPolygon = None
    for contour in contours:
        contourPerimeter = cv2.arcLength(contour, True)
        approximatedPolygon = cv2.approxPolyDP(contour, 0.06*contourPerimeter, closed=True)
        # Quadrilateral Detected
        if(len(approximatedPolygon) == 4):
            break
    return approximatedPolygon

# Draw the Approximated Polygon in Green on the Image
def drawLocalizedPlate(image, approximatedPolygon):
    M=cv2.moments(approximatedPolygon)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    
    finalImage = cv2.drawContours(image, [approximatedPolygon], -1, (0, 255, 0), 3)
    
    cv2.circle(finalImage, (cX, cY), 7, (0, 255, 0), -1)
    cv2.putText(finalImage, "Centroid of Plate: ("+str(cX)+", "+str(cY)+")", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return finalImage

def main(mode, number, imageName, steps, write):
    global IMAGE_LIST
    if(mode == "single"):
        IMAGE_LIST = [imageName]
    elif(mode == "multi"):
        IMAGE_LIST = IMAGE_LIST[:number]
    for image in IMAGE_LIST:
        try:
            # Read
            imagePath = os.path.join(DATASET_PATH, image)
            initialImage = readImage(imagePath)
            cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Original Image", initialImage)

            # Convert to Grayscale
            grayscaleImage = convertImageToGrayscale(initialImage)

            # Noise Removal
            noiseRemovedImage = noiseRemoval(grayscaleImage)

            # Histogram Equalization
            histEqImage = histogramEqualization(noiseRemovedImage)

            # Structuring Element for morphological opening
            openingStructElem = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # Morphological Opening
            openedImage = morphologicalOpening(histEqImage, openingStructElem)

            # Subtract Opened Image from HistEqImage
            subtractedImage = subtractOpenFromHistEq(histEqImage, openedImage)

            # Thresholding
            threshImage = thresholdingImage(subtractedImage)

            # Canny Edge Detection
            edgeDetectedImage = edgeDetection(threshImage, 250, 255)

            # Structuring Element for dilation
            dilationStructElem = np.ones((3, 3), np.uint8)
            # Dilation
            dilatedImage = imageDilation(edgeDetectedImage, dilationStructElem)

            # Finding the Plate        
            contours = findContours(dilatedImage)
            approximatedPolygon = approximateContours(contours)
            

            # Draw Plate Border on Original Image
            finalImage = drawLocalizedPlate(initialImage, approximatedPolygon)

            if(steps):
                cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Grayscale Image", grayscaleImage)

                cv2.namedWindow("Noise Removed Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Noise Removed Image", noiseRemovedImage)

                cv2.namedWindow("Histogram Equalized Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Histogram Equalized Image", histEqImage)

                cv2.namedWindow("Opened Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Opened Image", openedImage)

                cv2.namedWindow("Subtracted Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Subtracted Image", subtractedImage)

                cv2.namedWindow("Thresholded Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Thresholded Image", threshImage)

                cv2.namedWindow("Edge Detected Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Edge Detected Image", edgeDetectedImage)

                cv2.namedWindow("Dilated Edge Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Dilated Edge Image", dilatedImage)

            cv2.namedWindow("Final Output Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Final Output Image", initialImage)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # Save Output Image to FinalOutput directory
            if(write):
                if(steps):
                    os.chdir(OUTPUT_PATH)
                    os.makedirs(os.path.splitext(image)[0])
                    os.chdir('..')
                    STEPS = os.path.join(OUTPUT_PATH, os.path.splitext(image)[0])
                    cv2.imwrite(os.path.join(STEPS, '1.png'), grayscaleImage)
                    cv2.imwrite(os.path.join(STEPS, '2.png'), noiseRemovedImage)
                    cv2.imwrite(os.path.join(STEPS, '3.png'), histEqImage)
                    cv2.imwrite(os.path.join(STEPS, '4.png'), openedImage)
                    cv2.imwrite(os.path.join(STEPS, '5.png'), subtractedImage)
                    cv2.imwrite(os.path.join(STEPS, '6.png'), threshImage)
                    cv2.imwrite(os.path.join(STEPS, '7.png'), edgeDetectedImage)
                    cv2.imwrite(os.path.join(STEPS, '8.png'), dilatedImage)
                cv2.imwrite(os.path.join(OUTPUT_PATH, os.path.splitext(image)[0] + "-detected.png"), finalImage)
        except:
            print("Fatal Error. Cannot Continue.")

if __name__ == '__main__':
    imageName = None
    steps = False
    write = False
    number = 0
    mode = "--all"
    if("--write" in sys.argv):
        write = True
    if("--showsteps" in sys.argv):
        steps = True
    if("--single" in sys.argv):
        mode = "single"
        imageName = sys.argv[sys.argv.index("--single")+1]
        if(imageName not in IMAGE_LIST):
            print("Image not found in dataset. Exiting...")
            sys.exit(1)
    elif("--multi" in sys.argv):
        mode = "multi"
        try:
            number = int(sys.argv[sys.argv.index("--multi")+1])
        except:
            print("Invalid Input for number. Use valid number or --all instead.")            
        if(number > len(IMAGE_LIST)):
            print("Number entered exceeds number of images in dataset. Using --all argument instead...")
            number = len(IMAGE_LIST)
    main(mode, number, imageName, steps, write)
