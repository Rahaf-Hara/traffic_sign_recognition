from pprint import PrettyPrinter
import torch
import numpy as np
import json
from torchvision import transforms
from PIL import Image,ImageDraw,ImageFont
from collections import OrderedDict
import pandas as pd
from utilities import*
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import confusion_matrix
import time


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(15, 15)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%' % (p)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % (p)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)


def save_to_pickle(path, item):
    """
    Function that saves an item to pickle file
    """
    with open(path, mode='wb') as f:
        pickle.dump(item, f)


def load_from_pickle(path):
    """
    Function that loads an item from pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


device = "cuda"

all_labels = []
all_labels_df = pd.read_csv("./lisa-traffic-sign/allAnnotations.csv", sep=';')
all_labels_df['Annotation tag'].value_counts()
for_cm = list(all_labels_df['Annotation tag'].value_counts().keys())
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Data parameters
data_folder = './'  # folder with data files

# Load model checkpoint
checkpoint = './kaggle_checkpoints/checkpoint.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

with open('./kaggle_checkpoints/label_map.json', 'r') as j:
    label_to_int = json.load(j)

int_to_label = {v: k for k, v in label_to_int .items()}
# Transforms
resize = transforms.Resize((600, 600))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
all_results = {}
results = {}
for label in set(all_labels_df['Annotation tag']):
    results[label] = {}
    results[label]['true_positive'] = 0
    results[label]['false_positive'] = 0
    results[label]['false_negative'] = 0

all_results['true_positive'] = 0
all_results['false_positive'] = 0
all_results['false_negative'] = 0


def box_match(box_1, label_1, box_2, label_2, max_overlap=0.3):
    """
    Determines whether two boxes overlap themselves and have the same label
    :param box_1: the first box
    :param label_1: the label of the first box
    :param box_2: the second box
    :param label_2: the label of the second box
    :param max_overlap: threshold value for determining whether two boxes overlap themselves
    :return: boolean value indicating whether the boxes overlap themselves and have the same label
    """
    return find_jaccard_overlap(box_1.unsqueeze(0), box_2.unsqueeze(0)) > max_overlap and label_1 == label_2


def my_evaluate(img, original_image, img_id, annotations, cm_det_labels, cm_true_labels, min_score=0.45, max_overlap=0.3, top_k=200, annotate_image=True):
    """
    Detect objects in an image with a trained SSD600, and visualize the results.
    :param original_image: image, a PIL Image
    :param img_id: the identifier of the image, used as file name
    :param annotations: ground truth information on the traffic signs in the image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param annotate_image: boolean variable indicating whether annotated images shall be written to a file
    """

    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)
    # Forward prop. # Add a batch dimension:
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    # Detect objects in SSD output
    tic = time.time()
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)
    toc = time.time()
    inference_time = toc-tic

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [int_to_label[l] for l in det_labels[0].to('cpu').tolist()]
    det_scores = det_scores[0].to('cpu').tolist()

    # Annotate
    annotated_image = original_image

    if det_labels != ['background']:
        for i in range(det_boxes.size(0)):
   
            # Create an image showing the detected traffic signs, if requested
            if annotate_image:
                draw = ImageDraw.Draw(annotated_image)
                font = ImageFont.load_default()
                # Boxes
                box_location = det_boxes[i].tolist()
                draw.rectangle(xy=box_location, outline='#ff0000')
                draw.rectangle(xy=[l + 1. for l in box_location], outline='#ff0000')  # a second rectangle at an offset of 1 pixel to increase line thickness

                # Text
                text = det_labels[i].upper() + ' ' + str(det_scores[i])
                text_size = font.getsize(text)
                text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
                textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                    box_location[1]]
                draw.rectangle(xy=textbox_location, fill='#ff0000')
                draw.text(xy=text_location, text=text, fill='white', font=font)
                del draw

            # For every detection, see whether it matches a ground truth
            match = False
            for j in range(len(annotations['boxes'])):
                if annotations['labels'][j] == -1:
                    continue
                match = box_match(det_boxes[i],
                                  det_labels[i],
                                  torch.Tensor(annotations['boxes'][j]),
                                  int_to_label[annotations['labels'][j]])

                cm_det_labels.append(det_labels[i])
                cm_true_labels.append(int_to_label[annotations['labels'][j]])

                if match:
                    annotations['labels'][j] = -1
                    break

            if match:  # true positive if the detection is correct and matched a ground truth
                all_results['true_positive'] += 1
                results[det_labels[i]]['true_positive'] += 1
            else:  # false positive if the detection did not match a ground truth
                all_results['false_positive'] += 1
                results[det_labels[i]]['false_positive'] += 1

        if annotate_image:
            annotated_image.save('./' + img_id + '.png')

    # After all detections were checked, let us see whether the detector missed something
    for label in annotations['labels']:
        if label == -1:  # This is set after a detection matched this ground truth
            continue

        # false negative if we reach this line, since the ground truth object was not found
        results[int_to_label[label]]['false_negative'] += 1
        all_results['false_negative'] += 1
    return cm_det_labels, cm_true_labels, inference_time


def removeprefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


# Find IDs of images in the test data
with open(os.path.join('./kaggle_checkpoints', 'TEST'+ '_images.json'), 'r') as j:
    images = json.load(j)
with open(os.path.join('./kaggle_checkpoints', 'TEST' + '_objects.json'), 'r') as j:
    objects = json.load(j)

cm_det_labels = []
cm_true_labels= []
all_inf = []


if not os.path.exists("./predictions.pickle"):
    # from tqdm import tqdm
    # for obj,img in tqdm(zip(objects,images), total=len(images), desc="Images"):
    for i, (obj, img) in enumerate(zip(objects, images)):
        #     print(img)
        annotations = obj
        original_image = Image.open(img, mode='r')
        original_image = original_image.convert('RGB')
        b = img.find('_annotations/', 0)+13
        img_id = removeprefix(img, img[:b])
        img_id = re.sub('\.png$', '', img_id)
        allcm_det_label, allcm_true_label, inference_time = my_evaluate(img, original_image, img_id, annotations, cm_det_labels, cm_true_labels, annotate_image=True)
        all_inf.append(inference_time)
        if i % 100 == 0:
            print(i)

    # Calculate precisio and recall
    precision = {}
    recall = {}
    precision['ALL'] = all_results['true_positive'] / (all_results['true_positive'] + all_results['false_positive'])
    recall['ALL'] = all_results['true_positive'] / (all_results['true_positive'] + all_results['false_negative'])

    for label in set(all_labels_df['Annotation tag']):
        # Precision
        if results[label]['true_positive'] + results[label]['false_positive'] > 0:  # check for detections
            precision[label] = results[label]['true_positive'] / (results[label]['true_positive'] + results[label]['false_positive'])
        else:
            precision[label] = 'No detections'
            if results[label]['false_negative'] == 0:
                precision[label] = 'No detections, but also no signs in test set'

        # Recall
        if results[label]['true_positive'] + results[label]['false_negative'] > 0:  # check for ground truth objects
            recall[label] = results[label]['true_positive'] / (results[label]['true_positive'] + results[label]['false_negative'])
        else:
            recall[label] = 'No signs in test set'

    # Print results
    print('PRECISION')
    pp.pprint(precision)
    print()
    print('RECALL')
    pp.pprint(recall)

    save_to_pickle("./predictions.pickle", allcm_det_label)
    save_to_pickle("./groundtruth.pickle", allcm_true_label)

else:
    allcm_det_label = load_from_pickle("./predictions.pickle")
    allcm_true_label = load_from_pickle("./groundtruth.pickle")


print("Lists Lengths", len(allcm_det_label), len(allcm_true_label))

for_cm = [label_to_int[c] for c in for_cm]
print("before", for_cm)

allcm_det_label = [label_to_int[d] for d in allcm_det_label]
allcm_true_label = [label_to_int[t] for t in allcm_true_label]

for item in for_cm:
    if item not in allcm_true_label:
        for_cm.remove(item)

for item in for_cm:
    if item not in allcm_det_label:
        for_cm.remove(item)

print("after", for_cm)
print(allcm_det_label)
print()
print(allcm_true_label)
