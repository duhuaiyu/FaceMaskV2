
from bs4 import BeautifulSoup
import os
import shutil
from dataPrepare.util import *

source_dir = "/home/duhuaiyu/Downloads/facemaskdata/"
dest_dir = "/home/duhuaiyu/Downloads/facemaskdata/groung_truth"

# This program used to generate ground truth files used for evaluation
# output file format refer https://github.com/rafaelpadilla/Object-Detection-Metrics

def remove_folder_contents(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def generate_box(obj):

    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]
labels_array = ["nomask","mask","incorrect"]

def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        return target

if __name__ == '__main__':
    image_path = os.path.join(source_dir,"images")
    annotation_path = os.path.join(source_dir,"annotations")
    imgs = list(sorted(os.listdir(image_path)))
    labels = list(sorted(os.listdir(annotation_path)))


    for img_idx in range(750,853):

        file_name = "maksssksksss"+str(img_idx)
        label_path = os.path.join(annotation_path, file_name +".xml")
        img_path = os.path.join(image_path, file_name +".png")
        labels = generate_target(file_name,label_path)
        org_img = cv2.imread(img_path)


        process_chain = FlipHandler()
        process_chain.set_next(RotationHandler()).set_next(ChopHandler()).set_next(ResizeHandler())
        with open(os.path.join(dest_dir,file_name+".txt"), 'w') as the_file:
            for idx, box in enumerate(labels["boxes"]):
                classType = labels["labels"][idx]
                w = box[3] - box[1]
                h = box[2] - box[0]
                the_file.write('{0} {1} {2} {3} {4}\n'.format(labels_array[classType],box[0],box[1],box[2],box[3]))










