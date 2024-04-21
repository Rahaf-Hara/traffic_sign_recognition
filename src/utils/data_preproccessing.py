import json
import os
import torch
import random
from collections import OrderedDict
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare Label-to-int & Int-to-label Dictionaries
all_labels = []
all_labels_df = pd.read_csv("../input/lisa-traffic-sign/allAnnotations.csv", sep=';')
all_labels_df['Annotation tag'].value_counts()
# print(set(all_labels_df['Annotation tag']))
label_to_int = OrderedDict({label: num for num, label in enumerate(set(all_labels_df['Annotation tag']), start=1)})
label_to_int['background'] = 0
int_to_label = {v: k for k, v in label_to_int .items()}
int_to_label = OrderedDict(sorted(int_to_label.items(), key=lambda t: t[0]))


def train_valid_test(path_to_dataset, label_to_int):
    """
    split the dataset in the given path into three subsets(test,validation,train)
    :param path_to_dataset:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """
    # the initial entire dataset
    annotatios_file = glob.glob(os.path.join(path_to_dataset,'*tations.csv'))
    anno_data_frame = pd.read_csv(annotatios_file[0], sep=';')

    data_dict = {}
    for i, row in anno_data_frame.iterrows():

        file, label, Upper_left_corner_X, Upper_left_corner_Y, Lower_right_corner_X, Lower_right_corner_Y, *_ = row
        file = os.path.join(path_to_dataset, file)
        if file not in data_dict:
            boxes = list()
            labels = list()
            data_dict[file] = {"boxes": boxes, "labels": labels}
            boxes.append([Upper_left_corner_X, Upper_left_corner_Y, Lower_right_corner_X, Lower_right_corner_Y])
            labels.append(label_to_int[label])

            data_dict[file]["boxes"] = boxes
            data_dict[file]["labels"] = labels
        else:
            data_dict[file]["boxes"].append([Upper_left_corner_X, Upper_left_corner_Y, Lower_right_corner_X, Lower_right_corner_Y])
            data_dict[file]["labels"].append(label_to_int[label])

    X = list(data_dict.keys())
    y = list(data_dict.values())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=42)
    return X_train, X_test, y_train, y_test


X_train = list(); X_test = list(); Y_train = list(); Y_test = list()
sub = list()
directory = 'DATASET/'
for root, subdirectories, files in os.walk(directory):
    for subdirectory in subdirectories:
        if subdirectory.endswith('annotations'):
            x_train, x_test, y_train, y_test = train_valid_test(os.path.join(root, subdirectory), label_to_int)
            X_train.extend(x_train)
            X_test.extend(x_test)
            Y_train.extend(y_train)
            Y_test.extend(y_test)
            sub.append(subdirectory)

print(len(X_train), len(X_test))
# Train
output_folder = "RESOURCES"
with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
    json.dump(X_train, j)
with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
    json.dump(Y_train, j)
with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
    json.dump(label_to_int, j)  # save label map too
# Test
output_folder = "RESOURCES"
with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
    json.dump(X_test, j)
with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
    json.dump(Y_test, j)