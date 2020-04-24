# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
import cv2
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo3weights_new_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/classes.txt',
        "score": 0.5,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.graph = tf.get_default_graph()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        print(self.boxes)
        print(self.scores)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou, max_boxes=40)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()  # 计时
        if image is None:
            print('Open Error! Try again!')
            return
        print(image.shape)

        # print(image)
        # print(image.__class__)
        # print(image.shape) (height, width, channels)
        # print(image.dtype)  uint8
        # imread返回对象类型为<class 'numpy.ndarray'> 数组大小为(height, width, channels) 每个元素类型为<class 'numpy.uint8'>
        # 其中image[i][j]是一个三维数组，分别存放(i, j)BGR通道数据

        boxed_image = YOLO.letterbox_image(image, self.model_image_size)  # 缩放图片到合适的大小
        boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)
        image_data = boxed_image.astype(np.float32)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        with self.graph.as_default():
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.shape[0], image.shape[1]],
                    K.learning_phase(): 0
                })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 300
        thickness -= thickness % 2  # 保证为偶数
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
                label_origin = (left - thickness // 2, top - label_size[1] - baseline)
                label_end = (label_origin[0] + label_size[0], top)
            else:
                label_origin = (left - thickness // 2, top + 1)
                label_end = (label_origin[0] + label_size[0], label_origin[1] + label_size[1] + baseline)

            text_origin = (left, label_end[1] - baseline)  # putText文字以左下角为origin
            # 在图片中标注检测项目
            image = cv2.rectangle(image, (left, top),
                                  (right, bottom), self.colors[c], thickness)
            # image = cv2.rectangle(image, label_origin, label_end, self.colors[c], -1)
            # image = cv2.putText(image, label, text_origin,
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        end = timer()  # 计时
        print(end - start)
        return image

    @staticmethod
    def letterbox_image(image, size):
        """不改变图像长宽比，用padding填充，缩放image到size尺寸"""
        ih = image.shape[0]
        iw = image.shape[1]
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = cv2.resize(image, (nw, nh))
        image = cv2.copyMakeBorder(image, (h - nh) // 2, h - nh - (h - nh) // 2, (w - nw) //
                                   2, w - nw - (w - nw) // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image

    def close_session(self):
        self.sess.close()