from PIL import Image
import pytesseract
import argparse
import cv2
import os


#constructing arguements to parse through the code
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the input image readable by tesseract OCR")
ap.add_argument("-p","--preprocessing",type=str,default="thresh",help="type of preprocessing done")
args=vars(ap.parse_args())

#loading the image onto the code
image=cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#check if arguement passed needs to apply thresholding
if args["preprocessing"]=="thresh":
   gray=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
# check if blurring is required to reduce noise
elif args["preprocessing"]=="blur":
    gray=cv2.medianBlur(gray,3)
# save grayscale image to system as temp file to apply OCR to it (tmp file will have name that the subroutine has assigned it)
file="{}.png".format(os.getpid())
cv2.imwrite(file,gray)

#load the image in PIL format, then apply OCR to it, finally delete the temp file
text=pytesseract.image_to_string(Image.open(file))
os.remove(file)
print(text)

#showing the output image
cv2.imshow("Image",image)
cv2.imshow("Output",gray)
cv2.waitKey(0)