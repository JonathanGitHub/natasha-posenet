import numpy as np
import cv2 as cv
from PIL import Image
from pose_engine import PoseEngine

def main():

    cap = cv.VideoCapture(0)
    engine = PoseEngine('models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    i = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # TESTING CODE

        im = Image.fromarray(frame)
        poses, inf_time = engine.DetectPosesInImage(im)

        print(f"Inf time: {inf_time}")
        
        for pose in poses:
            # if pose.score < 0.4: continue
            print('\nPose Score: ', pose.score)
            for label, keypoint in pose.keypoints.items():
                print('  %-20s x=%-4d y=%-4d score=%.1f' %
                    (label, keypoint.point[0], keypoint.point[1], keypoint.score))

        # Display the resulting frame
        cv.imshow('frame', frame)

        # Exit the frame with 'q' press
        if cv.waitKey(1) == ord('q'):
            break

        print(i)
        i+=1
        
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()