import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4)
parser.add_argument('--imageSize', type=tuple, default=[256,512])
parser.add_argument('--out_ch', type=int, default=1, help='size of the latent z vector')
parser.add_argument('--last_ckpt', type=str, default="logs/checkpoint/ckpt_best_37.pth", help='continue the training process')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--resume', default=True, help='folder to output images and model checkpoints')
opt = parser.parse_args()
