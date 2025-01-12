import torch
import torch.nn as nn
import numpy as np
import cv2


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:400, 0:288]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def visualize_image_opencv(image_data):
    # ensure the image is 2D
    image_data_2d = image_data.squeeze()

    # display using opencv
    cv2.imshow('Processed Image', image_data_2d)
    cv2.waitKey(0)  # wait for key press
    cv2.destroyAllWindows()
