#!/usr/bin/env python2
# coding: utf-8

import numpy as np
import cv2
import math
import Image
import math

def prepare_batch(image_files, image_mean, batch_size):
    num_images = len(image_files)

    IMAGE_DIM = 256
    CROPPED_DIM = 227
    indices = [1, IMAGE_DIM-CROPPED_DIM+1]
    center = int(math.floor(indices[1] / 2))

    num_images = len(image_files)
    #images = np.zeros((CROPPED_DIM,CROPPED_DIM,3,batch_size), dtype=float)
    images = np.zeros((batch_size, 3, CROPPED_DIM, CROPPED_DIM), dtype=float)

    #image_mean = np.transpose(image_mean,(2, 1, 0))

    for i in range(num_images):
        # read file
        # fprintf('%c Preparing %s\n',13,image_files{i});
        im = cv2.imread(image_files[i]).astype(float)
        # resize to fixed input size
        im = cv2.resize(im, (IMAGE_DIM, IMAGE_DIM), cv2.INTER_LINEAR) # i.e. Bilinear interpolation
        # im = cv2.resize(im, (IMAGE_DIM, IMAGE_DIM), cv2.INTER_CUBIC) # still not OK https://stackoverflow.com/questions/26812289/matlab-vs-c-vs-opencv-imresize/Resize
        # im = cv2.resize(im, (IMAGE_DIM, IMAGE_DIM), cv2.INTER_AREA) # still not OK
        # im = cv2.resize(im, (IMAGE_DIM, IMAGE_DIM), cv2.INTER_NEAREST) # still not OK
        # im = cv2.resize(im, (IMAGE_DIM, IMAGE_DIM), cv2.INTER_LANCZOS4) # still not OK
        im = np.round(im.astype(float), 4)
        """
        im = Image.open(image_files[i])
        im = im.resize((IMAGE_DIM, IMAGE_DIM), Image.ANTIALIAS)
        im = np.array(im)
        im = im.astype(np.float32)
        """
        
        # Transform GRAY to RGB
        #if size(im,3) == 1
        #    im = cat(3,im,im,im)
        #end
        # permute from RGB to BGR (IMAGE_MEAN is already BGR)
        # im = im(:,:,[3 2 1]) - image_mean
        
        im = im - image_mean
        im = np.transpose(im, (2, 0, 1))
        #im = (np.ceil(im * 10000)/10000.0).astype(np.float32)
        # im = np.int32(np.ceil(im*100000+5))/100000.0
        # Crop the center of the image
        #images(:,:,:,i) = permute(im(center:center+CROPPED_DIM-1,...
        #    center:center+CROPPED_DIM-1,:),[2 1 3])

        images[i,:,:,:] = im[:, center:center+CROPPED_DIM, center:center+CROPPED_DIM]
    return images