import imutils 

# here the image is put as input
# with step size telling how many pixels to skip in the image while sliding
# ws shows the size in height and width of the window to be slide in the image
def sliding_window(image,step,ws):
    #sliding a window across image 
    for y in range(0,image.shape[0]-ws[1],step):
        for x in range(0,image.shape[1]-ws[0],step):
            #yield current window
            yield(x,y,image[y:y+ws[1],x:x+ws[0]])

# here the image pyramid is built with a default scale reduction of 1.5
# minSize of the image is also defaulted, both can be overridden
def image_pyramid(image,scale=1.5,minSize=(224,224)):
    # yielding the current image
    yield(image)
    # looping over the original image to form pyramid
    while True:
        # computing dimensions of next image in the pyramid
        w=int(image.shape[1]/scale)
        image=imutils.resize(image,width=w)
        
        # if resized image doesn't meet supplied minimum size the stop making pyramid
        if image.shape[0]<minSize[1] or image.shape[1]<minSize[0]:
            break
        # yield the next image in the pyramid
        yield(image)
