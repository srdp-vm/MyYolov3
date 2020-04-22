import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import colorsys
import os
from timeit import default_timer as timer
from yolo3.model import yolo_eval, yolo_body


class YOLO(object):

    def __init__(self):
        self.classes_path = "model_data/classes.txt"
        self.anchors_path = "model_data/yolo_anchors.txt"
        self.model_path = "model_data/trained.h5"
        self.model_image_size = (416, 416)
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = tf.Session(config=config)
        # K.tensorflow_backend.set_session(session)
        self.sess = K.get_session()
        self.yolo_model = None
        self.colors = None
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]  # 固定hsv模型中的s，v为1，h从0-1变化
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(
            lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))  # 颜色转化为RGB 0-255
        np.random.seed(10001)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        # print(colors)
        self.input_image_shape = None
        self.boxes, self.scores, self.classes = self.generate()
    
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        # print(class_names)
        return class_names
    
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        # print(anchors)
        return anchors
    
    def generate(self):
        # load model
        self.model_path = os.path.expanduser(self.model_path)  # 路径转换为windows格式
        assert self.model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(self.model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)),
                                   num_anchors//3, num_classes)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        print("generate finished")
        # print(boxes)   #Tensor("concat_11:0", shape=(?, 4), dtype=float32)
        # print(scores)  #Tensor("concat_12:0", shape=(?,), dtype=float32)
        # print(classes) #Tensor("concat_13:0", shape=(?,), dtype=int32)
        # print(input_image_shape) #Tensor("Placeholder_366:0", shape=(2,), dtype=float32)
        # print(boxes.__class__)   #<class 'tensorflow.python.framework.ops.Tensor'>
        return boxes, scores, classes

    @staticmethod
    def letterbox_image(image, size):
        """不改变图像长宽比，用padding填充，缩放image到size尺寸"""
        ih = image.shape[0]
        iw = image.shape[1]
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = cv2.resize(image, (nw, nh))
        image = cv2.copyMakeBorder(image, (h-nh)//2, h-nh-(h-nh)//2, (w-nw) //
                                   2, w-nw-(w-nw)//2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image

    def detect_image(self, image):
        start = timer()  # 计时
        if image is None:
            print('Open Error! Try again!')
            return
        print(image.shape)
    
        # print(image)
        # print(image.__class__)
        #print(image.shape) (height, width, channels)
        # print(image.dtype)  uint8
        # imread返回对象类型为<class 'numpy.ndarray'> 数组大小为(height, width, channels) 每个元素类型为<class 'numpy.uint8'>
        # 其中image[i][j]是一个三维数组，分别存放(i, j)BGR通道数据
    
        boxed_image = YOLO.letterbox_image(image, self.model_image_size)  # 缩放图片到合适的大小
        boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)
        image_data = boxed_image.astype(np.float32)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.


        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
    
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    
        thickness = (image.shape[0] + image.shape[1]) // 300
        thickness -= thickness % 2   # 保证为偶数
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
    
            label = '{} {:.2f}'.format(predicted_class, score)
    
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
    
            print(label, (left, top), (right, bottom))
    
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            if top - label_size[1] >= 0:
                label_origin = (left-thickness//2, top - label_size[1] - baseline)
                label_end = (label_origin[0]+label_size[0], top)
            else:
                label_origin = (left-thickness//2, top + 1)
                label_end = (label_origin[0]+label_size[0], label_origin[1]+label_size[1]+baseline)
    
            text_origin = (left, label_end[1]-baseline)  # putText文字以左下角为origin
            # 在图片中标注检测项目
            image = cv2.rectangle(image, (left, top),
                                  (right, bottom), self.colors[c],  )
            image = cv2.rectangle(image, label_origin, label_end, self.colors[c], -1)
            image = cv2.putText(image, label, text_origin,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
        end = timer()  # 计时
        print(end - start)
        return image
    
    def close_session(self):
        self.sess.close()