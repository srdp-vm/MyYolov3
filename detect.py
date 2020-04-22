import cv2
import os
import argparse
from yolo import YOLO

def detectImage() :
    yolo = YOLO()
    while True :
        path = input("Input filename:")
        if path == "exit" :
            break
        image = cv2.imread(path)
        if image is None :
            print('Open Error! Try again!')
            continue
        result = yolo.detect_image(image)
        if not result is None :
            cv2.imshow("Result", result)
            cv2.waitKey()
            cv2.destroyAllWindows()
    print("Input finish")
    yolo.close_session()

def detectFolder(path, outputPath = "output"):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    yolo = YOLO()
    names = os.listdir(path)
    for name in names:
        filepath = os.path.join(path, name)
        image = cv2.imread(filepath)
        if image is None:
            print("Open file {} error".format(path))
            break
        result = yolo.detect_image(image)
        if not result is None:
            output = os.path.join(outputPath, name)
            print(output)
            cv2.imwrite(output, result)
    yolo.close_session()


def detectVideo(path, output_path = ""):
    yolo = YOLO()
    if path.isnumeric():
        cap = cv2.VideoCapture(int(path))
    else :
        cap = cv2.VideoCapture(path)
    if not cap.isOpened() :
        raise IOError("Couldn't open camera or video")
    video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    while True:
        # get a frame
        ret, frame = cap.read()
        if not ret :
            print("finish to read capture")
        # show a frame
        result = yolo.detect_image(frame)
        cv2.imshow("capture", result)
        if cv2.waitKey(1) > 0:
            break
        if isOutput:
            out.write(result)
    cap.release()
    cv2.destroyAllWindows()
    print("finish to read capture")
    yolo.close_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--image', default=False, action="store_true",
            help='Image detection mode, will ignore all positional arguments')
    parser.add_argument("--folder", type=str, default=None, help="Folder path to detect")
    parser.add_argument("--video", type=str, default=None, help='Path for video to detect or index for camera')
    parser.add_argument("--output", type=str, default=None, help='Path to save the result video')

    argvs = parser.parse_args()

    if argvs.image:
        detectImage()
    elif argvs.video:
        if argvs.video != "" :
            detectVideo(argvs.video, argvs.output)
        else:
            print("Please input video path!")
    elif argvs.folder:
        if argvs.folder != "":
            detectFolder(argvs.folder, "output")
        else:
            print("Please input video path!")
    else:
        print("Please input at least one argument")