# this will be the model used for classifying images
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
# after applying sliding window concept, this supression will remove weak detections
from imutils.object_detection import non_max_suppression
# these helpers will provide functions for sliding window
from Helper.detection_helper import sliding_window
from Helper.detection_helper import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#constructing arguements for command line
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
    help="path to the input image")
ap.add_argument("-s","--size",type=str,default="(200,150)",
    help="ROI size input (in pixels)")
ap.add_argument("-c","--min-conf",type=float,default=0.9,
    help="minimum probability to filter weak detections")
#extra step
ap.add_argument("-v","--visualize",type=int,default=-1,
    help="whether or not to show extra visualizations for debugging")
args=vars(ap.parse_args())

# initializing variables for object detection procedure
#the following data is to be implemented for width of image
# pyramid scaling and window step size of the image
WIDTH=600
PYR_SCALE=1.5
WIN_STEP=16
ROI_SIZE=eval(args["size"])
INPUT_SIZE=(224,224)

#loading the network
print("[INFO] loading network...")
model=ResNet50(weights="imagenet",include_top=True)

#loading input image and resizing it to the supplied width
orig=cv2.imread(args["image"])
orig=imutils.resize(orig,width=WIDTH)
(H,W)=orig.shape[:2]

#intializing the image_pyramid 
pyramid=image_pyramid(orig,scale=PYR_SCALE,minSize=ROI_SIZE)

#initializing ROIS and Locations
rois=[] #stores ROI of pyramid and sliding windows
locs=[] #stores x-y coordinates of ROI in the actual image

#time taking to process sliding in pyramid layers
start=time.time()

#looping over the image pyramid
for image in pyramid:
    # determining the scale factor between original dimensions and current layer of pyramid
    scale=W/float(image.shape[1])

    #for each layer of image pyramid, looping over sliding window locations
    for (x,y,roiOrig) in sliding_window(image,WIN_STEP,ROI_SIZE):
        # scaling the (x,y) coordinates of ROI with respect to original image dimensions
        x=int(x*scale)
        y=int(y*scale)
        w=int(ROI_SIZE[0]*scale)
        h=int(ROI_SIZE[1]*scale)
        
        # use the ROI and preprocess it to classify image using tensorflow
        roi=cv2.resize(roiOrig,INPUT_SIZE)
        roi=img_to_array(roi)
        roi=preprocess_input(roi)

        # update the ROI list and associated coordinates
        rois.append(roi)
        locs.append((x,y,x+w,y+h))

        # check to see if we visualize each of sliding window in image pyramid
        if args["visualize"]>0:
            # clone original image and bound it with current region
            clone=orig.copy()
            cv2.rectangle(clone,(x,y),(x+w,y+h),
                (0,255,0),2)
            #show the visualization and current ROI
            cv2.imshow("Visualization",clone)
            cv2.imshow("ROI",roiOrig)
            cv2.waitKey(0)
# showing time taken to scan through the pyramid 
end=time.time()
print("[INFO] looping over pyramid and windows took {:.5f} seconds".format(end-start))

#convert the ROIs to a Numpy array
rois=np.array(rois,dtype="float32")

# classifying each proposal ROIs using ResNet and showing how long it took
print("[INFO] classifying ROIs...")
start=time.time()
preds=model.predict(rois)
end=time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(end-start))

# decoding predictions and initializing dict to map labels which were recognized
preds=imagenet_utils.decode_predictions(preds,top=1)
labels={}

#loop over the predictions
for (i,p) in enumerate(preds):
    # take prediction information for current ROI
    (imagenetID,label,prob)=p[0]

    #filtering out weak predictions
    if prob>=args["min_conf"]:
        #assign bounding box with predictions and convert to coordinates
        box=locs[i]
        #take list of predictions for label and add bounding box with probability to list
        L=labels.get(label,[])
        L.append((box,prob))
        labels[label]=L
clone=None
# looping over the labels for each detected objects
for label in labels.keys():
    # clone the original image to draw bounding box on it
    print("[INFO] showing results for '{}'".format(label))
    clone=orig.copy()

    #loop over all bounding boxes for current label
    for (box,prob) in labels[label]:
        # drawing bounding box
        (startX,startY,endX,endY)=box
        cv2.rectangle(clone,(startX,startY),(endX,endY),
            (0,255,0),2)
        # showing results before and after applying non_maxima suppression
        cv2.imshow("Before",clone)
        clone=orig.copy()
        # extract bounding boxes and associated predictions and apply non-maxima suppression
        boxes=np.array([p[0] for p in labels[label]])
        proba=np.array([p[1] for p in labels[label]])
        boxes=non_max_suppression(boxes,proba)

        # looping over all bounding boxes after applying non-maxia suppression
        for (startX,startY,endX,endY) in boxes:
            # drawing bounding box on image
            cv2.rectangle(clone,(startX,startY),(endX,endY),
                (0,255,0),2)
            y=startY-10 if startY-10>10 else startY+10
            cv2.putText(clone,label,(startX,y),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
        
    # showing output after supression
    cv2.imshow("After",clone)
    cv2.waitKey(0)