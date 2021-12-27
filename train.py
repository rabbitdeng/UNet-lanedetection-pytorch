import os
import torch
import argparse
from torchvision.transforms import transforms
from model import UNet
from dataset import TusimpleDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import parser
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    #transforms.Resize(opt.imageSize),

])


def train():
    model = UNet(in_ch=1, out_ch=opt.out_ch).to(device)
    start_epoch = -1
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=opt.lr)
    train_dataset = TusimpleDataset(opt.data_path, transform=x_transform, target_transform=y_transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
    if opt.resume:
        path_checkpoint =opt.last_ckpt  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    dataset_size = len(train_dataset)
    x_axis = []
    loss_axis = []
    for epoch in range(start_epoch + 1, opt.epochs):
        step = 0
        running_loss = 0.0
        pbar = tqdm(train_loader)
        for inputs, masks in pbar:
            inputs, masks = inputs.to(device), masks.to(device)
            masks = masks.float()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description("Epoch:[%d/ %d] Average running loss %.7f" % (epoch,
                                                                               opt.epochs,
                                                                               (running_loss / dataset_size)))
            step += 1
            # print("%d / %d, step loss: %0.5f" % (step, dataset_size // batch_size, loss.item()))
        print("Epoch: %d ---running loss : %0.5f" % (epoch, running_loss))
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        if not os.path.isdir("./logs/checkpoint"):
            os.mkdir("./logs/checkpoint")
        torch.save(checkpoint, './logs/checkpoint/ckpt_best_%s.pth' % (str(epoch)))
        x_axis.append(float(epoch))
        loss_axis.append(float(running_loss))
        if not os.path.isdir("./logs"):
            os.mkdir("./logs")
        #plt.figure(1)
        plt.title("loss")
        plt.plot(x_axis, loss_axis)
        plt.savefig("loss.png")
    plt.show()


if __name__ == '__main__':
    train()
