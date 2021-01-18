import torch
import argparse
from models import *













if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='weights/0_prune_300epoch_0.0001s/yolov4_org.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/VOC_Infrared/Ultralytics/infrared.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/0_prune_300epoch_0.0001s/last.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.4, help='global channel prune percent')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    percent = opt.percent
    # 指定GPU
    torch.cuda.set_device(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg).to(device)

    print(model.module_list[0][1].parameter)

