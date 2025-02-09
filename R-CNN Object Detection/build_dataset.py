from Setup.iou import compute_iou
from Setup import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os
import time

start=time.time()
# looping over positives and negatives in directories
for dirPath in (config.positive_path,config.negative_path):
    # if path doesn't exist, create it
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

# grab all image paths in input images directory
imagePaths=list(paths.list_images(config.orig_images))

# initialize total no. of positives and negatives
totalPositive=0
totalNegative=0

# looping over image paths
for (i,imagePath) in enumerate(imagePaths):
    # showing number of images processed
    print("[INFO] preprocessing image {}/{}".format(i+1,len(imagePaths)))

    # extracting image filename from file path and using it to derive XML file path
    filename=imagePath.split(os.path.sep)[-1]
    filename=filename[:filename.rfind(".")]
    annotPath=os.path.sep.join([config.orig_annots,"{}.xml".format(filename)])

    # loading the annotations file,
    # building soup to pull data from XML files
    contents=open(annotPath).read()
    soup=BeautifulSoup(contents,"html.parser")
    # making ground truth boxes for bounding box rectangles
    gtBoxes=[]

    # extracting image dimensions
    w=int(soup.find("width").string)
    h=int(soup.find("height").string)
    
    #looping over all "object" elements
    for o in soup.find_all("object"):
        # extracting labels and bounding box coordinates
        label=o.find("name").string
        xMin=int(o.find("xmin").string)
        yMin=int(o.find("ymin").string)
        xMax=int(o.find("ymax").string)
        yMax=int(o.find("ymax").string)

        # truncate any bounding box coordinates 
        # that fall outside boundaries of image
        xMin=max(0,xMin)
        yMin=max(0,yMin)
        xMax=max(w,xMax)
        yMax=max(h,yMax)

        # updating the list of ground truth bounding boxes
        gtBoxes.append((xMin,yMin,xMax,yMax))
    
    # loading input image
    image=cv2.imread(imagePath)

    # running selective search on the image
    # initializing list of proposed boxes
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects=ss.process()
    proposedRects=[]

    # looping over the rectangles generated by the selective search
    for (x,y,w,h) in rects:
        # converting bounding boxes to startX,startY,endX,endY
        proposedRects.append((x,y,x+w,y+h))
    
    #initilizing counters for counting positive and negative ROis saved
    positiveROIs=0
    negativeROIs=0

    #looping over the max region proposals
    for proposedRect in proposedRects[:config.max_prop]:
        # extract the rectangle bounding box
        (propStartX,propStartY,propEndX,propEndY)=proposedRect
        # looping over ground-truth bounding boxes
        for gtBox in gtBoxes:
            # computing IOU
            iou=compute_iou(gtBox,proposedRect)
            (gtStartX,gtStartY,gtEndX,gtEndY)=gtBox
            # initializing ROI and output path
            roi=None
            outputPath=None
            # check to see if IOU is greater than 0.7 and positive ROI less than the max limit
            if iou>0.7 and positiveROIs<=config.max_pos:
                # extract the ROI and derive output path to psoitive instance
                roi=image[propStartY:propEndY,propStartX:propEndX]
                filename="{}.png".format(totalPositive)
                outputPath=os.path.sep.join([config.positive_path,filename])
                # increment the positive counter
                positiveROIs+=1
                totalPositive+=1
			# determining if proposed box falls within the ground truth box
            fullOverlap=propStartX>=gtStartX
            fullOverlap=fullOverlap and propStartY>=gtStartY
            fullOverlap=fullOverlap and propEndX<=gtEndX
            fullOverlap=fullOverlap and propEndY<=gtEndY

            # check to see if there is not full overlap and IOU is less than 0.05
            # check also, if negative counter limit is not reached
            if not fullOverlap and iou<0.05 and negativeROIs<=config.max_neg:
                # extract ROI and derive output path to negative instance
                roi=image[propStartY:propEndY,propStartX:propEndX]
                filename="{}.png".format(totalNegative)
                outputPath=os.path.sep.join([config.negative_path,filename])

                # incrementing the negative counters
                negativeROIs+=1
                totalNegative+=1
			# check if both ROIs and output path are valid
            if roi is not None and outputPath is not None:
                # resize ROI to input dimensions of CNN
                roi=cv2.resize(roi,config.input_dims,interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath,roi)
end=time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("time taken is {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))