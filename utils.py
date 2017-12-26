# -*- coding: utf-8 -*-
import skimage
import skimage.io
import skimage.transform
import numpy as np


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path,img_size=224,float_flag=True): # 임의의 크기 이미지를 받아들여, 244 x 244로 변형해 준다.
    # load image
    img = skimage.io.imread(path)  # numpy.ndarray. M x N x 3 
    if float_flag == True:
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    if img.ndim ==3:
        if img.shape[2] > 3:
            img = img[:,:,:3]
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    if float_flag == True:
        resized_img = skimage.transform.resize(crop_img, (img_size, img_size))
    else:
        #resized_img = skimage.transform.resize(crop_img, (img_size, img_size),preserve_range=True)  # 정수지만, type은 float.용량 많이 차리함. 실행 속도도 느림.
        resized_img = (skimage.transform.resize(crop_img, (img_size, img_size)) * 255).astype(np.int)   #  실행 솓고 빠름
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
