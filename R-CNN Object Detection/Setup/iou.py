def compute_iou(boxA,boxB):
    # determining x,y coordinates of intersection triangle
    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=max(boxA[2],boxB[2])
    yB=max(boxA[3],boxB[3])

    # computing area of intersection of rectangle
    interArea=max(0,xA-xB+1)*max(0,yB-yA+1)
    # compute area of both prediction and ground-truth rectangles
    boxAArea=(boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxBArea=(boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    # computing iou using intersection area divided by sum of prediction + ground truth areas - intersection areas
    iou=interArea/float(boxAArea+boxBArea-interArea)
    # returning iou
    return iou
    