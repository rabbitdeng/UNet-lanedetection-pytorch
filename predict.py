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

opt = parser.parse_args()

x_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=opt.out_ch),
    transforms.Resize(opt.imageSize),
    # transforms.Grayscale(num_output_channels=1),
    # 标准化至[-1,1],规定均值和标准差
    transforms.Normalize([0.5], [0.5]),  # torchvision.transforms.Normalize(mean, std, inplace=False)

])
# mask只需要转换为tensor
y_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    # transforms.Resize(opt.imageSize),

])

test_dataset = TusimpleDataset(opt.data_path, transform=x_transform, target_transform=y_transform)

test_loader = DataLoader(dataset=test_dataset, batch_size=1)

device = torch.device('cpu')


def test():
    model = UNet(1, 1)
    path_checkpoint = "./logs/checkpoint/ckpt_best_20.pth"  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点

    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    model.eval()
    for inputs, _ in test_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)

        outputs = outputs.squeeze(0)
        inputs = inputs.squeeze(0)
        outputs = outputs.squeeze(0)
        inputs = inputs.squeeze(0)
        outputs = torch.cat((outputs, inputs), 1)
        img_y = outputs.detach().numpy()
        img_y = Image.fromarray(img_y * 255).convert("L")
        # PIL_image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
        img_y.show()


if __name__ == '__main__':
    test()
