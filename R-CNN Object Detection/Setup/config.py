import os

# defining path for the original input images
# defining path for the annotations on the particular image

orig_base_path="raccoons"
orig_images=os.path.sep.join([orig_base_path,"images"])
orig_annots=os.path.sep.join([orig_base_path,"annotations"])

#path required for storing the dataset formed after running
# selective search on the images 
base_path="dataset"
positive_path=os.path.sep.join([base_path,"raccoon"])
negative_path=os.path.sep.join([base_path,"no_raccoon"])

# defining no. of max proposals for selective search
# for gathering training data and performing inference
max_prop=2000
max_prop_infer=200

# defining max positives and negatives in each image
max_pos=30
max_neg=10

# initializing image input dimensions for the network
input_dims=(224,224)

# defining path to output model and label binarizer
model_path="raccoon_detector.h5"
encoder_path="label_encoder.pickle"

# minimum probability required for a positive prediction
min_proba=0.99