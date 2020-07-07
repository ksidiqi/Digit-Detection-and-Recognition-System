import cv2
import logging
import numpy as np
from torchvision import transforms
from torchvision import models
import torchvision

from scipy.io import loadmat

import torch
import torch.nn as nn
from PIL import Image

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import os

import h5py
import tqdm
from collections import defaultdict
import copy

DEBUG = True
SAMPLE = 10 if DEBUG else 10000000000


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# classes
classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9', 'none')


class Util:

    @staticmethod
    def get_img_pyramid(img, level=5):
        logging.debug("pyramind size={}, level={}", img.size, level)
        pyramid = [img]
        for _ in range(level):
            pyramid.append(cv2.pyrUp(pyramid[-1]))
        return pyramid

    @staticmethod
    def de_noise(img):
        # https://docs.opencv.org/
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return dst

    @staticmethod
    def sliding_window(img, size=(32, 32), step=8):
        logging.debug("sliding window img={}, size={}, step={}", img.size, size, step)

        def window_iter():
            h, w = img.shape[:2]
            for i in range(0, h - size[0], step):
                for j in range(0, w - size[1], step):
                    logging.debug("[{}:{}, {}:{}]", i, i + size[0], j, j + size[1])
                    iend, jend = min(i + size[0], h), min(j + size[1], w)
                    yield (img[i:iend, j:jend], np.array((i, j, iend, jend)))

        windows = []
        for w in window_iter():
            windows.append(w)
        return windows

    @staticmethod
    def padd_with_zeros(img, new_size):
        new_img = np.zeros(new_size, dtype=img.dtype)
        new_img[0:img.shape[0], 0:img.shape[1]] = img
        return new_img

    def fast_forward(self):
        pass


class CustomModel(nn.Module):

    def __init__(self, num_classes=11, init_weights=True):
        super(CustomModel, self).__init__()
        layers = [
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), #added extra layers
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes) #limit the number of classes
        )

    def forward(self, x):
        #used pytorch forward configs
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class Optimizer:

    def __init__(self, model, lr=0.001, mom=0.9):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)


class TransformerFactory:

    @staticmethod
    def get_transformer(model):
        preprocess = {
            'train':
                transforms.Compose([
                    transforms.Resize(128),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
            'val':
                transforms.Compose([
                    transforms.Resize(128),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        }
        return preprocess


import heapq
SEARCHING, RECORDING = 0, 1
class ModelTrainEval:

    def __init__(self, optimizer: Optimizer, ep=1):
        self.model = optimizer.model
        self.criterion = optimizer.criterion
        self.optimizer = optimizer.optimizer
        self.ep = ep

    def eval(self, trainloader, testloader):
        print("training")

        pq = [0, 0, 0, 0, 0, 0, 0]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        model.to(device)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(self.ep):
            running_loss = 0.0
            running_corrects = 0
            total = 0
            model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                total += len(preds)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / total
            epoch_acc = 100 * running_corrects.double() / total

            #val
            running_loss = 0.0
            running_corrects = 0
            total = 0
            model.eval()
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                total += len(preds)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            val_epoch_loss = running_loss / total
            val_epoch_acc = 100 * running_corrects.double() / total

            if val_epoch_acc > best_acc:  # update best
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = val_epoch_acc
            print(str.format("{},{},{},{},{}", epoch, epoch_loss, epoch_acc.item(), val_epoch_loss, val_epoch_acc.item()))
            if val_epoch_acc.item() <= heapq.heappop(pq):
                break
            heapq.heappush(pq, val_epoch_acc.item())

        model.load_state_dict(best_model_wts)
        print('Finished Training')

    def predict(self, dataloader, limit=4):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        model.eval()
        mm = defaultdict(list)
        for data in dataloader:
            image, rec, lay = data
            image = image.to(device)
            outputs = model(image)
            #sm = nn.Softmax(dim=1)
            #outputs = sm(outputs)
            prob, predicted = torch.max(outputs), torch.argmax(outputs)
            record = mm[lay.item()]
            if not record:
                record.extend([[], [], []])
            record[0].append(tuple(element.item() for element in rec.flatten()))
            record[1].append(prob.item())
            record[2].append(predicted.item())
        bbox, scores, digits = [],[], []
        for layer in sorted(mm.keys(), reverse=False):
            bbox.extend(mm[layer][0])
            scores.extend(mm[layer][1])
            digits.extend(mm[layer][2])

        tensor_box, scores = torch.DoubleTensor(bbox) , torch.DoubleTensor(scores)
        found = torchvision.ops.nms(tensor_box, scores, iou_threshold=0.95)
        best_bbox = []
        for i in found.flatten():
            idx = i.item()
            best_bbox.append((bbox[idx], digits[idx]))
            if len(best_bbox) == limit:
                break
        return best_bbox




class ModelFactory:

    @staticmethod
    def modify_last_layer(model, num_classes=11, freeze=True):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    @staticmethod
    def create(model):
        if model == "custom":
            model = CustomModel()
        elif model == "vgg16":
            model = models.vgg16(pretrained=False)
            ModelFactory.modify_last_layer(model, freeze=False)
        elif model == "vgg16_weighted":
            model = models.vgg16(pretrained=True)
            ModelFactory.modify_last_layer(model, freeze=True)
        return model


class CustomDataSet(Dataset):

    def __init__(self, mat_file, root_dir, transform=None):
        self.street_address_frames = loadmat(os.path.join(root_dir, mat_file))
        self.root_dir = root_dir
        self.transform = transform
        self.images = self.street_address_frames['X']
        self.labels = self.street_address_frames['y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = Image.fromarray(self.images[:, :, :, idx], 'RGB')
        labels = torch.tensor(np.squeeze(self.labels[idx]), dtype=torch.long)
        if self.transform:
            images = self.transform(images)
        return images, labels


class ImageDataSet(Dataset):

    # get_box_data and get_name copied from
    # https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
    @staticmethod
    def get_box_data(index, hdf5_data):
        """
        get `left, top, width, height` of each picture
        :param index:
        :param hdf5_data:
        :return:
        """
        meta_data = dict()
        meta_data['height'] = []
        meta_data['label'] = []
        meta_data['left'] = []
        meta_data['top'] = []
        meta_data['width'] = []

        def print_attrs(name, obj):
            vals = []
            if obj.shape[0] == 1:
                vals.append(obj[0][0])
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(hdf5_data[obj[k][0]][0][0]))
            meta_data[name] = vals

        box = hdf5_data['/digitStruct/bbox'][index]
        hdf5_data[box[0]].visititems(print_attrs)
        return meta_data

    @staticmethod
    def get_name(index, hdf5_data):
        name = hdf5_data['/digitStruct/name']
        return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][:]])

    def __init__(self, mat_file, root_dir, transform=None):
        file = h5py.File(os.path.join(root_dir, mat_file), 'r')
        size = file['/digitStruct/name'].size
        images = []
        for _i in tqdm.tqdm(range(size)):
            pic = ImageDataSet.get_name(_i, file)
            box = ImageDataSet.get_box_data(_i, file)
            for i in range(len(box['label'])):
                tmpdict = {key: box[key][i] for key in box.keys()}
                tmpdict['name'] = pic
                images.append(tmpdict)
            if _i == SAMPLE:
                break
        self.root_dir = root_dir
        self.transform = transform
        self.struct = images
        self.mem = {}  # cache loaded image

    def __len__(self):
        return len(self.struct)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.struct[idx]['name'])
        image = self.mem.get(img_name, Image.open(img_name))
        self.mem[img_name] = image
        left, top = self.struct[idx]['left'], self.struct[idx]['top']
        w = int(self.struct[idx]['width'])
        h = int(self.struct[idx]['height'])
        a, b = 0, 0
        #image.show()
        image = image.crop((left - a, top-b, left + w+ a, top + h+b))
        #image.show()
        image = self.transform(image)
        #tmp = transforms.ToPILImage()(image)
        #tmp.show(title=str(self.struct[idx]['label']))
        label = torch.tensor(self.struct[idx]['label'], dtype=torch.long)
        return image, label


class VideoDataSet(Dataset):

    def __init__(self, video_file, root_dir, transform=None):
        cap = cv2.VideoCapture(os.path.join(video_file, root_dir))
        frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(color)
        cap.release()
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        #cv2.imshow(str(idx), frame)
        #cv2.waitKey(1000)
        image = Image.fromarray(frame, 'RGB')
        image = self.transform(image)
        return image

class PyrimadDataSet(Dataset):

    def __init__(self, fname, transform):
        windows = []
        img = cv2.imread(fname)
        tmp = np.indices((img.shape[0], img.shape[1]))
        tmp = np.rollaxis(tmp, 0, 3)
        pyup = Util.get_img_pyramid(img, 3)
        for i in range(3):
            level = pyup[i]
            tmp = cv2.resize(tmp, (level.shape[1], level.shape[0]), interpolation=cv2.INTER_NEAREST)
            for window, box in Util.sliding_window(level, (300, 160), 80):
                yx1, yx2 = tmp[box[0], box[1]], tmp[box[2], box[3]]
                org_box = np.array([yx1[1], yx1[0], yx2[1], yx2[0]])
                windows.append((window, org_box, i))
                #cv2.imshow("", window)
                #cv2.waitKey(100)
        self.windows = windows
        self.transform = transform


    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        image = Image.fromarray(self.windows[idx][0], 'RGB')
        image = self.transform(image)
        return image, self.windows[idx][1], self.windows[idx][2]




print("cuda:0" if torch.cuda.is_available() else "cpu")


def train_and_test_all_model():
    for model_name in ['vgg16_weighted',  'vgg16', 'custom']:
        print("running " + model_name)
        mat, train_dir, test_dir = "digitStruct.mat", "./data/train", "./data/test"
        dataset = ImageDataSet(mat, train_dir, TransformerFactory.get_transformer(model_name)['train'])
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16)
        test_dataset = ImageDataSet(mat, test_dir, TransformerFactory.get_transformer(model_name)['val'])
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=16)
        model = ModelFactory.create(model_name)
        optimizer = Optimizer(model)
        model_train_eval = ModelTrainEval(optimizer, ep=50)
        model_train_eval.eval(dataloader, test_dataloader)
        torch.save(model.state_dict(), './model/'+model_name+'.pt')

def load_model_and_predict(root, root_out, model_name, files):
    model = ModelFactory.create(model_name)
    model.load_state_dict(torch.load(str.format("./model/{}.pt", model_name)))
    model_train_eval = ModelTrainEval(Optimizer(model), 0)
    for f in files:
        img_file_name = os.path.join(root, f)
        pyrimad_data_set = PyrimadDataSet(img_file_name, TransformerFactory.get_transformer("")['val'])
        window_data_loader = DataLoader(pyrimad_data_set, batch_size=1, num_workers=0)
        digits = model_train_eval.predict(window_data_loader)
        orgimg = cv2.imread(img_file_name)
        for box, digit in digits:
            cv2.rectangle(orgimg, (box[0], box[1]), (box[2], box[3]), color=255, thickness=1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(orgimg, str(digit), (box[2], box[3]), font, 2, (0, 0, 255), 2, cv2.QT_FONT_BOLD)
        print(os.path.join(root_out, f))
        cv2.imwrite(os.path.join(root_out, f), orgimg)


train_and_test_all_model()
load_model_and_predict("./input", "./output", "vgg16", ["1.jpg", '2.jpg', "3.jpg", "4.jpg", "5.jpg"])
