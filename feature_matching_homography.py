# -*- coding: utf-8 -*-
# taken from
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
#
import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 20

def drawMatches(img1, kp1, img2, kp2, matches, outfile="pic_out.jpg", color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
#    r = 15
#     thickness = 2
#    if color:
#        c = color
#    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
#        if not color:
#            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
#        try:
#            end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
#            end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
#            cv2.line(new_img, end1, end2, c, thickness)
#            cv2.circle(new_img, end1, r, c, thickness)
#            cv2.circle(new_img, end2, r, c, thickness)
#        except:
#            print m.trainIdx, " -- ", m.queryIdx

    # Original
    plt.figure(figsize=(15,15))
    plt.imshow(new_img, 'gray')
    #plt.show()
    plt.savefig(outfile)

    # AK
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(range(100))

    #fig.savefig("graph.png")

base_dir = "/home/mosenturm/mtg"
catalog_dir = "/katalog"

#img1 = cv2.imread(base_dir + '/turn_against.orig.jpg',0) # queryImage from catalog

img2 = cv2.imread(base_dir + '/turn_against_sparko.jpg',0) # trainImage from camera scan

# we need to keep in mind aspect ratio so the image does
# not look skewed or distorted -- therefore, we calculate
# the ratio of the new image to the old image
r = 600.0 / img2.shape[1]
dim = (600, int(img2.shape[0] * r))

# perform the actual resizing of the image and show it
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

#rotate img2 by 180°
irows,icols = img2.shape
iM = cv2.getRotationMatrix2D((icols/2,irows/2),180,1)
img2 = cv2.warpAffine(img2,iM,(icols,irows))

# lookup the catalog for card pictures
# loop starts in catalog dir
for dir, subdirs, fnames in os.walk(base_dir + catalog_dir):
    print dir
    print subdirs
    print fnames
    for fname in fnames:
        match_found = False
        path = os.path.join(dir, fname)
        #img = cv.LoadImage(path.encode('utf-8'),0)
        try:
        	img1 = cv2.imread(path,0) # queryImage from catalog
        except IOError:
        	print >> sys.stderr, "\tLoading image %s failed" % (path)
        	next

        # Initiate SIFT detector
        sift = cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2_org = img2
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)
            print "Enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            match_found = True
            outfile = base_dir + '/match_' + fname

            # Sort the matches based on distance.  Least distance
            # is better
            #print vars(matches)
            good = sorted(good, key=lambda val: val.distance)
            print good[0].distance
            print good[1].distance
            print good[3].distance

            drawMatches(img1,kp1,img2_org,kp2,good[:50], outfile, color = (0,255,0))
        else:
            img2_org = img2
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None
            outfile = base_dir + '/nomatch_' + fname


        #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #                   singlePointColor = None,
        #                   matchesMask = matchesMask, # draw only inliers
        #                   flags = 2)

        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)


        # reset img2 to original
        img2 = img2_org


