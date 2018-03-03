import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from CDNet2014Dataset import CDNet2014Dataset, Rescale, ToTensor
from model1 import BackSubModel1
from data_utils import count_labels_distribution
import cv2


def test_vid(vid_path, chk):

    # Instantiate model
    model = BackSubModel1()

    if torch.cuda.is_available():
        print('Using GPU:', torch.cuda.get_device_name(0))

        model.cuda()
    else:
        print('NO GPU DETECTED!')

    print('Loading checkpoint ...')
    model.load_state_dict(torch.load(chk))

    cap = cv2.VideoCapture(os.path.join(vid_path, 'video.mp4'))
    bg = cv2.imread(os.path.join(vid_path, 'background.jpg'))
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg = cv2.resize(bg, (320, 240))

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate foreground
        fg = cv2.absdiff(frame, bg)
        # set threshold to ignore small differences
        _, fg = cv2.threshold(fg, 50, 255, 0)

        # concatenate image, background, and foreground
        img_input = cv2.merge([frame, bg, fg])
        img_input = img_input.astype(np.float32) / 255

        img_input = img_input.transpose((2, 0, 1))
        img_input = Variable(torch.from_numpy(np.expand_dims(img_input, 0)),
                             volatile = True).cuda()

        # Forward pass only to get logits/output
        outputs = model(img_input)
        
        # Get predictions from the maximum value
        _, prediction = torch.max(outputs.data, 1)

        prediction = prediction.cpu().numpy()
        prediction = np.squeeze(prediction)
        prediction = prediction.astype(np.float32)
        print('output size:', prediction.shape)
        
        cv2.imshow('Video', frame)
        cv2.imshow('Pred', prediction)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == "__main__":
    chk_path = '/home2/backsub_repo/checkpoints/model3d/model1_camerajitter.pkl'
    vid_path = '/datasets/backsub/cdnet2014/dataset/cameraJitter/badminton'

    test_vid(vid_path, chk_path)
