import numpy as np
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
    help="path to the input image")
ap.add_argument("-p","--prototxt",required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m","--model",required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c","--confidence",type=float,default=0.2,
    help="minimum probability to filter weak detections")
#returning the arguements into a dictionary
args=vars(ap.parse_args())

#list of classes that can be detected
CLASSES=["background","aeroplane","bicycle","bird","boat",
         "bottle","bus","car","cat","chair","cow","diningtable",
         "dog","horse","motorbike","person","pottedplant","sheep",
         "sofa","train","tvmonitor"]
#generate individual random colors for each of the classes
COLORS=np.random.uniform(0,255,size=(len(CLASSES),3))

#loading the model
print("[INFO] loading model...")
net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

#reading the input image and constructing input blob for it
image=cv2.imread(args["image"])
(h,w)=image.shape[:2]
blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007843,
    (300,300),127.5)

#passing blob through the network and obtaining and printing predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections=net.forward()

#looping over detections
for i in np.arange(0,detections.shape[2]):
    #extracting the probability associated with prediction
    confidence=detections[0,0,i,2]
    #filtering out weak detections where confidence threshold is not satified
    if confidence>args["confidence"]:
        #extracting the index of class label from detections
        #computing x,y coordinates of the bounding box
        idx=int(detections[0,0,i,1])
        box=detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY)=box.astype("int")

        #displaying predictions
        label="{}:{:.2f}%".format(CLASSES[idx],confidence*100)
        print("[INFO] {}".format(label))
        cv2.rectangle(image,(startX,startY),(endX,endY),
            COLORS[idx],2)
        #coordinates to put label
        y=startY-15 if startY-15>15 else startY+15
        cv2.putText(image,label,(startX,y),
        cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
cv2.imshow("Output",image)
cv2.waitKey(0)