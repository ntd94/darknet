from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import my_darknet
import sys

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def cvDrawBoxes(detections, img, dw, dh):
    for detection in detections:
        # if human
        if detection[0] != b'person':
            continue
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]

        x = x * dw
        y = y * dh
        w = w * dw
        h = h * dh
        
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    # configPath = "/home/dat/source/hello_darknet_python/csresnext50-panet-spp-original-optimal.cfg"
    # weightPath = "/home/dat/source/hello_darknet_python/csresnext50-panet-spp-original-optimal_final.weights"
    # configPath = "/home/dat/source/hello_darknet_python/enet-coco.cfg"
    # weightPath = "/home/dat/source/hello_darknet_python/enetb0-coco_final.weights"
    configPath = "/media/dat/05C830EB6380925C/yolo-setups/yolo-setup/yolov3.cfg"
    weightPath = "/media/dat/05C830EB6380925C/yolo-setups/yolo-setup/yolov3.weights"
    metaPath = "/home/dat/Downloads/AlexeyAB-darknet/cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = my_darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = my_darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('rtsp://QDT:QdtVtx@2020@10.61.166.15/profile3/media.smp', cv2.CAP_FFMPEG)
    if cap is None:
        print("ERROR reading stream")
        sys.exit(-1)
    cap.set(cv2.CAP_PROP_FPS, 3)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # cap.set(3, 1280)
    # cap.set(4, 720)
    # out = cv2.VideoWriter(
    #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for store device image
    device_image = my_darknet.allocate_mems(int(h), int(w), 3, 0)

    # Create an image we reuse for each detect
    darknet_image = my_darknet.allocate_mems(my_darknet.network_height(netMain),
                                            my_darknet.network_width(netMain),
                                            3, 1)
    dw = w / my_darknet.network_width(netMain)
    dh = h / my_darknet.network_height(netMain)

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if ret != True or frame_read is None:
            break

        # frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(frame_read,
        #                            (my_darknet.network_width(netMain),
        #                             my_darknet.network_height(netMain)),
        #                            interpolation=cv2.INTER_LINEAR)

        # darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        indata = frame_read.ctypes.data_as(POINTER(c_ubyte))
        bla = cast(device_image.data, c_void_p)
        my_darknet.copyHostToDevice(device_image.data, indata, int(3*h*w))
        # indata = frame_read.ctypes.data_as(c_char_p)

        my_darknet.preprocessRGB(cast(device_image.data, POINTER(c_ubyte)), int(h), int(w), darknet_image.data, darknet_image.h, darknet_image.w)

        # detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        detections = my_darknet.detect_image_custom(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_read, dw, dh)
        print()
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        if 27 == cv2.waitKey():
            break
    cap.release()
    # out.release()

if __name__ == "__main__":
    YOLO()

