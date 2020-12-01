import argparse
import cv2
import random
import time

#constructing arguements
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the input image")
ap.add_argument("-m","--method",type=str,default="fast",choices=["fast","quality"],help="selective search method")
args=vars(ap.parse_args())

# loading the input image
image=cv2.imread(args["image"])
# showing original image
cv2.imshow("Initial",image)

# initialize OpenCV's selective search implementation and set the
# input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# checking which algorithm to use ('fast','quality')
if args["method"]=="fast":
    print("[INFO] using *fast* selective search")
    ss.switchToSelectiveSearchFast()
else:
    print("[INFO] using *quality* selective search")
    ss.switchToSelectiveSearchQuality()

#running selective search on input image
start=time.time()
rects=ss.process()
end=time.time()

#time taken for selective search to work
# number of regions proposed
print("[INFO] selective search took {:.4f} seconds".format(end-start))
print("[INFO] {} total region proposals".format(len(rects)))

#looping and visualizing the proposals
for i in range(0,len(rects),100):
    #cloning the original image to allowing drawing on it
    output=image.copy()

    # looping over the subset of region proposals
    for (x,y,w,h) in rects[i:i+100]:
        # drawing a region proposal bounding box on the image
        color=[random.randint(0,255) for j in range(0,3)]
        cv2.rectangle(output,(x,y),(x+w,y+h),color,2)
    #showing the image
    cv2.imshow("output",output)
    key=cv2.waitKey(0) & 0xFF
    if key==ord("q"):
        break