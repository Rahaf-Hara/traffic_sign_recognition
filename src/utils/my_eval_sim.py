import torch
from itertools import product as product
from  utils.utilities import *
from PIL import ImageDraw,ImageFont


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

def my_evaluate(device,model,int_to_label,normalize,resize,to_tensor,original_image, min_score=0.45, max_overlap=0.3, top_k=200, annotate_image=True):
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

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [int_to_label[l] for l in det_labels[0].to('cpu').tolist()]
    det_scores = det_scores[0].to('cpu').tolist()
    print("det_boxes.size(0)",det_boxes[0][0])
    # Anntate
    annotated_image = original_image
    dic = {}
    if det_labels != ['background']:
        for i in range(det_boxes.size(0)):
            box_location = det_boxes[i].tolist()
            dic[det_labels[i]] = box_location
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

                annotated_image.save('RESOURCES' + 'perception' + '.png')

        return dic
