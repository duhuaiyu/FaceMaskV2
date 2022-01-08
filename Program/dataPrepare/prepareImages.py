
from bs4 import BeautifulSoup
import os
import shutil

from dataPrepare.util import *

source_dir = "/home/duhuaiyu/Downloads/facemaskdata/"
#dest_dir = "/home/duhuaiyu/Downloads/facemaskdata/classification_data_new_s"
dest_dir = "/home/duhuaiyu/Downloads/facemaskdata/temp"
resize_dim=(32,32)
catagory = ["train","validation","test"]
dirs = ["nomask","mask","incorrect","background"]
generate_file_count = [15,5,15,40]
#generate_file_count = [1,1,1,1]
random_background = 1
background_size = [20, 40, 60, 80, 100, 120, 150]
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
def create_rotation (img,num, dir,name):

    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    for i in range(num):
        degree = random.randint(-30,30)
        M = cv2.getRotationMatrix2D(center, degree, scale  =1.2)
        rotated = cv2.warpAffine(img, M, (w, h))
        save_path = os.path.join(dest_dir,dir ,name+"_"+str(i)+".png")
        cv2.imwrite(save_path,rotated)

if __name__ == '__main__':
    image_path = os.path.join(source_dir,"images")
    annotation_path = os.path.join(source_dir,"annotations")
    imgs = list(sorted(os.listdir(image_path)))
    labels = list(sorted(os.listdir(annotation_path)))

    # clear output folder
    for cate in catagory:
        for dir in dirs:
            remove_folder_contents(os.path.join(dest_dir,cate,dir))

    for img_idx, img in enumerate(imgs):
        head, tail = os.path.split(img)
        file_name = tail[:-4]
        label_path = os.path.join(annotation_path, file_name +".xml")
        img_path = os.path.join(image_path, file_name +".png")
        labels = generate_target(file_name,label_path)
        org_img = cv2.imread(img_path)
        print(labels)
        if img_idx < 600:
            dest_dir_sub = os.path.join(dest_dir,"train")
        elif img_idx < 750:
            dest_dir_sub = os.path.join(dest_dir,"validation")
        else:
            dest_dir_sub = os.path.join(dest_dir,"test")

        process_chain = FlipHandler()
        process_chain.set_next(RotationHandler()).set_next(ChopHandler()).set_next(ResizeHandler())

        image_shape = org_img.shape
        for idx, box in enumerate(labels["boxes"]):
            classType = labels["labels"][idx]
            w = box[3] - box[1]
            h = box[2] - box[0]
            ori_size_img = org_img[box[1]:box[3],box[0]:box[2]]

            save_path = os.path.join(dest_dir_sub, dirs[classType] , file_name+str(idx)+".png")
            cv2.imwrite(save_path, cv2.resize(ori_size_img,resize_dim))
            # validation and test set do not generate more similar data
            if img_idx >= 600:
                continue
            # if ground truth is too small, do not generate more data
            if w < 10 and h < 10 :
                continue
            add_w = int(0.1*w)
            add_h = int(0.1 * h)
            new_box = [
                max(box[0] -add_h,0),
                max(box[1] - add_w, 0),
                min(box[2] + add_h, image_shape[1] + 1),
                min(box[3] + add_w, image_shape[0]+ 1)
            ]
            print(new_box)
            for i in range(generate_file_count[classType]):

                save_path = os.path.join(dest_dir_sub, dirs[classType], file_name +str(idx)+"_"+ str(i)+".png")
                new_img = org_img[new_box[1]:new_box[3],new_box[0]:new_box[2]]
                print(new_img.shape)
                res = process_chain.handle(new_img)
                print(res.shape)
                cv2.imwrite(save_path,res)

        i = 0
        while i < generate_file_count[3]:
            size_index = random.randint(0,len(background_size)-1)
            size = background_size[size_index]
            x = random.randint(0,image_shape[1]-size)
            y = random.randint(0,image_shape[0]-size)
            background_box = [x , y , x + size, y + size]
            overlap = False
            for idx, box in enumerate(labels["boxes"]):
                iou = get_IoU(background_box, box)
                print("background_box",background_box,"box",box,"iou",iou)
                if iou> 0.2:
                    print("roi > 0.2")
                    overlap = True
                    continue
            if overlap:
                continue
            new_img = org_img[y:y+size,x:x+size]
            resized = cv2.resize(new_img, resize_dim, interpolation = cv2.INTER_CUBIC)
            print(file_name+"_"+str(i)+".png")
            save_path = os.path.join(dest_dir_sub,dirs[3],file_name+"_"+str(i)+".png")
            cv2.imwrite(save_path,resized)
            i = i+1









