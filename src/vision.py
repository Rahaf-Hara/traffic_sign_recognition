#!/usr/bin/python3

import torch
from torchvision import  transforms
import pandas as pd
from itertools import product as product
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from utils.model import*
import torch
import rospy
from utils.my_eval_sim import my_evaluate
import cv2
from cv_bridge import CvBridge, CvBridgeError
import json


class Prediction:
    def __init__(self):

        self.frames_counter = 0
        self.bridge = CvBridge()
        # SUBSCRIBERS
        self.image_sub = rospy.Subscriber('/turtlebot/realsense_d435i/color/image_raw', Image,self.get_image)
        self.dic = {}

    def get_image(self, image):

        self.frames_counter += 1
        if self.frames_counter == 6:
            print("frame counter=6")
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
                cv_image = cv2.resize(cv_image, (600, 600))
                print('new image received')
            except CvBridgeError as e:
                print(e)
            im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.prediction(im_rgb)

    def prediction(self, image):

        all_labels_df = pd.read_csv("DATASET/allAnnotations.csv", sep=';')
        all_labels_df['Annotation tag'].value_counts()
        with open('RESOURCES/label_map.json', 'r') as j:
            label_to_int = json.load(j)

        int_to_label = {v: k for k, v in label_to_int .items()}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = SSD600(len(label_to_int))
        # Load model checkpoint
        checkpoint = 'RESOURCES/checkpoint.pth.tar'
        checkpoint = torch.load(checkpoint, map_location=device)
        state_dict = checkpoint['model'].state_dict()
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Transforms
        resize = transforms.Resize((600, 600))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Evaluate and annotate
        original_image = PILImage.fromarray(image)
        self.dic = my_evaluate(device, model, int_to_label, normalize, resize, to_tensor, original_image, annotate_image=True)


# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('Vision_node')
    node = Prediction()
    rospy.spin()
