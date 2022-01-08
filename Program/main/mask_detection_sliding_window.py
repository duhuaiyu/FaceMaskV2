import os.path
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
from dataPrepare.prepareImages import *
from dataPrepare.util import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MaskDetector:
    # labels used in final output image
    labels = {
        0: 'background',
        1: 'incorrect',
        2: 'mask',
        3: 'nomask',
    }
    # label and ROI colors for different classes
    colors = {
        0: (255, 255, 255),
        1: (255, 0, 0),  # incorrect
        2: (0, 255, 0),  # mask
        3: (0, 0, 255),  # nomask
    }
    # predefined window size and stride
    windwos = [
        {'size': [24, 24], 'stride': 4},
        {'size': [32, 32], 'stride': 5},
        {'size': [40, 40], 'stride': 6},
        {'size': [60, 60], 'stride': 10},
        {'size': [70, 70], 'stride': 10},
        {'size': [85, 85], 'stride': 10},
        {'size': [100, 100], 'stride': 13},
        {'size': [110, 110], 'stride': 13},
        {'size': [150, 150], 'stride': 13},
        {'size': [200, 200], 'stride': 20},
    ]
    input_size = (32, 32)

    def __init__(self, model_path):
        self.model_path = model_path
        # load saved model
        model = models.vgg16(pretrained=False)
        model.avgpool = Identity()
        model.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        self.model = model

    # chop images from image according to window size and stride
    @staticmethod
    def get_images_by_window(image, window):
        res = []
        rois = []
        h, w, _ = image.shape
        window_h = window['size'][0]
        window_w = window['size'][1]
        for i in range(0, h - window_h, window['stride']):
            for j in range(0, w - window_w, window['stride']):
                roi = (j, i, j + window_w, i + window_h)
                sub_img = image[roi[1]:roi[3], roi[0]:roi[2]]
                sub_img = cv2.resize(sub_img, MaskDetector.input_size)
                res.append(MaskDetector.process_image(sub_img))
                rois.append(roi)
        return res, rois

    # need to process img the same way we train the model
    @staticmethod
    def process_image(img):
        img = img / 255.
        mean = np.array([0.485, 0.456, 0.406])  # provided mean
        std = np.array([0.229, 0.224, 0.225])  # provided std
        img = (img - mean) / std
        img = img.transpose((2, 0, 1))
        return img

    @staticmethod
    def get_all_windows(image):
        all_images = []
        all_rois = []
        for window in MaskDetector.windwos:
            imgs, rois = MaskDetector.get_images_by_window(image, window)
            all_images.extend(imgs)
            all_rois.extend(rois)
        return all_images, all_rois

    def evaluate(self, imgs, rois):
        input = torch.tensor(np.array(imgs), dtype=torch.float)
        rois = torch.tensor(np.array(rois), dtype=torch.float)
        my_dataset = TensorDataset(input)  # create your datset
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataloaders = DataLoader(my_dataset, batch_size=100)
        scores_all = torch.tensor([])
        preds_all = torch.tensor([])
        dataiter = iter(dataloaders)
        # make prediction in batch
        with torch.no_grad():
            for batch_input in dataiter:
                inputs = batch_input[0].to(device)
                outputs = self.model(inputs)
                outputs = torch.softmax(outputs, 1)
                outputs = outputs.to("cpu")
                scores, preds = torch.max(outputs, 1)
                scores_all = torch.cat((scores_all, scores), 0)
                preds_all = torch.cat((preds_all, preds), 0)

        # eliminate all backgrounds
        indices = (preds_all != 0).nonzero(as_tuple=False)
        indices = indices.squeeze()

        filtered_socors = torch.index_select(scores_all, 0, indices)
        filtered_rois = torch.index_select(rois, 0, indices)
        filtered_preds = torch.index_select(preds_all, 0, indices)
        # use non-maximum suppression to eliminate overlapping ROIs
        res_indices = torchvision.ops.batched_nms(filtered_rois, filtered_socors, torch.ones_like(filtered_socors), 0.1)

        # keep socors, rois predis after non-maximum suppression
        filtered_socors = torch.index_select(filtered_socors, 0, res_indices)
        filtered_rois = torch.index_select(filtered_rois, 0, res_indices)
        filtered_preds = torch.index_select(filtered_preds, 0, res_indices)

        return filtered_socors, filtered_rois.int(), filtered_preds


data_dir = "/home/duhuaiyu/Downloads/facemaskdata/images"
label_data_dir = "/home/duhuaiyu/Downloads/facemaskdata/annotations"
base_out_put_dir = "/home/duhuaiyu/Downloads/facemaskdata/"
FP_output = "/home/duhuaiyu/Downloads/facemaskdata/FP_output"
perdiction_output_dir = "/home/duhuaiyu/Downloads/facemaskdata/prediction_files"
if not os.path.exists(FP_output):
    os.makedirs(FP_output)
# save False Positive sample or not
save_FP = False


# save False Positive samples
def create_FP_images(imgFileName, ori_img, rois):
    xml_filename = imgFileName[:-4] + ".xml"
    label_path = os.path.join(label_data_dir, xml_filename)
    labels = generate_target(imgFileName, label_path)
    for idx, roi in enumerate(rois):
        if is_FP(roi, labels):
            fileName = os.path.join(FP_output, "tf_" + imgFileName[:-4] + "_" + str(idx) + ".png")
            # cv2.imshow("1",ori_img[roi[1]:roi[3],roi[0]:roi[2]])
            # cv2.waitKey(0)
            cv2.imwrite(fileName, ori_img[roi[1]:roi[3], roi[0]:roi[2]])


# save False Positive samples
def is_FP(roi, labels):
    for idx, box in enumerate(labels["boxes"]):
        iou = get_IoU(roi, box)
        if iou > 0.3:
            return False
    return True

# predict a single image
def handle_image(fileName, out_put_dir, maskDetector):
    print(fileName)
    file_path = os.path.join(data_dir, fileName)
    img_BRG = cv2.imread(file_path)
    img = cv2.cvtColor(img_BRG, cv2.COLOR_BGR2RGB)
    imgs, rois = MaskDetector.get_all_windows(img)

    final_socors, final_rois, final_preds = maskDetector.evaluate(imgs, rois)
    save_output(img_BRG, final_socors, final_rois, final_preds, out_put_dir, fileName)
    if save_FP:
        create_FP_images(fileName, img_BRG, final_rois)
    imgs = None
    rois = None
    img_BRG = None
    img = None
    # print(final_rois)

# output file with ROIs and labels
def save_output(orignal_image, final_socors, final_rois, final_preds, out_put_dir, fileName):
    imgHeight, imgWidth, _ = orignal_image.shape
    thick = 1
    newImage = orignal_image.copy()
    with open(os.path.join(perdiction_output_dir, fileName[:-4] + ".txt"), 'w') as the_file:
        for i, score in enumerate(final_socors):
            if score < 0.8:
                break
            roi = final_rois[i].cpu().data.numpy().astype(int)
            pred = final_preds[i].item()
            cv2.rectangle(newImage, (roi[0], roi[1]), (roi[2], roi[3]), MaskDetector.colors[pred], thick)
            cv2.putText(newImage, MaskDetector.labels[pred] + ":" + '{:.2f}'.format(score),
                        (roi[0], roi[1] - 12), 0, 1e-3 * imgHeight, MaskDetector.colors[pred], thick // 3)
            # generate prediction output files used for evaluation
            the_file.write(
                "{0} {1} {2} {3} {4} {5}\n".format(MaskDetector.labels[pred], score, roi[0], roi[1], roi[2], roi[3]))
    output_file = os.path.join(out_put_dir, fileName)
    cv2.imwrite(output_file, newImage)


if __name__ == '__main__':
    # out_put_dir = os.path.join(base_out_put_dir, "output_"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    out_put_dir = os.path.join(base_out_put_dir, "output_1_5")
    maskDetector = MaskDetector('../train'
                                '/checkpoint_1_1.pth')
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)
    for i in range(751, 853):
        handle_image("maksssksksss" + str(i) + ".png", out_put_dir, maskDetector)
