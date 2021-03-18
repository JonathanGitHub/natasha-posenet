import argparse
import cv2
import os
import numpy as np

from PIL import Image
from pose_engine import PoseEngine

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# LOOK TO
# https://github.com/google-coral/examples-camera/blob/master/opencv/detect.py

# NEXT TIME 
# IMPLEMENT THIS ONTO ALIEN INVASION
# PROJECT STICK CHARACTER ONTO THE IMAGE
# STICK DATA TO ALIEN INVASION

def main():
    
    # None of these arguments are needed
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    
    args = parser.parse_args()

    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model)

    cap = cv2.VideoCapture(args.camera_idx)

    while cap.isOpened():

        # Capture frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # image flip (0: x-axis, 1: y-axis, -1: both axis)
        # image is flipped on default 
        frame = cv2.flip(frame, 1)
        im = Image.fromarray(frame)
        poses, _ = engine.DetectPosesInImage(im)

        print("INFERENCE")

        for pose in poses:
            # if pose.score < 0.4: continue
            print('\nPose Score: ', pose.score)
            for label, keypoint in pose.keypoints.items():
                print('  %-20s x=%-4d y=%-4d score=%.1f' %
                    (label, keypoint.point[0], keypoint.point[1], keypoint.score))

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit the frame with 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()