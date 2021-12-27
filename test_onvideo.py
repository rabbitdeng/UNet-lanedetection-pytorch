import cv2
import torch
from torchvision.transforms import transforms
from model import UNet
from dataset import TusimpleDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import parser
import PIL.Image as Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_ad = "D:/lane_d_ts/test_video/IMG_1455.mp4"
output_ad = "test_output/write_test.avi"
video = cv2.VideoCapture(input_ad)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
image_size = (512, 256)
out = cv2.VideoWriter(output_ad, fourcc, 30.0, (512, 256))
frame_count = 0
model = UNet(1, 1).to(device=device)
path_checkpoint = "./logs/checkpoint/ckpt_best_20.pth"  # 断点路径
checkpoint = torch.load(path_checkpoint)  # 加载断点

model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
fps = video.get(cv2.cv2.CAP_PROP_FPS)
print("Original video fps: %d;" % fps)
while True:
    ret, img = video.read()
    if not ret:
        break
    else:
        if cv2.waitKey(30) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            print("writing finishing!")
            break
        frame_count += 1

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)

        image = image.transpose(0, 1)  # transform to pytorch CHW form
        image = image[np.newaxis, :, :]
        image = image / 255
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0)
        segmentation = model(image.cuda())

        binary_mask = segmentation.data.cpu().numpy()
        binary_mask = binary_mask * 255
        binary_mask = binary_mask.squeeze()
        # binary_mask = binary_mask.transpose(1, 0)
        binary_mask = binary_mask.astype(np.uint8)
        for elem in np.nditer(binary_mask):
            if elem < 128:
                elem = 0
            else:
                elem = 255
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        out.write(binary_mask)

output = cv2.VideoCapture(output_ad)
fps = output.get(cv2.cv2.CAP_PROP_FPS)
print("Original video fps: %d;" % fps)
cv2.destroyAllWindows()
