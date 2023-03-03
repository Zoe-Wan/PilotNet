import argparse
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt

from config import get_cfg_defaults
from model import build_model
from util.logger import *
from util.save_image import save_image

import os
import torch
import csv
import numpy as np
import cv2


class DrivingDatasetDataset(object):
    def __init__(self, cfg, data_dir='../../driving_dataset', ann_path='../../driving_dataset/data.txt'):
        self.cfg = cfg
        self.data_dir = os.path.join(data_dir, './data')
        with open(ann_path, 'r') as ann_file:
            ann_reader = csv.reader(ann_file, delimiter=' ')
            self.annotations = dict([r for r in ann_reader])

        self.mean_ann = np.mean([abs(float(val)) for key, val in self.annotations.items()])
        self.id_to_filename = dict([(i, key) for i, (key, val) in enumerate(self.annotations.items())])

    def __getitem__(self, idx):
        steering_command = float(self.annotations[self.id_to_filename[idx]])
        filepath = os.path.join(self.data_dir, self.id_to_filename[idx])
        image = self._preprocess_img(cv2.imread(filepath))
        image = torch.from_numpy(image)
        steering_command = torch.tensor([steering_command])

        return image, steering_command

    def __len__(self):
        return len(self.annotations)

    def _preprocess_img(self, img):
        # set training images resized shape
        h, w = self.cfg.IMAGE.TARGET_HEIGHT, self.cfg.IMAGE.TARGET_WIDTH

        # crop image (remove useless information)
        img_cropped = img[range(*self.cfg.IMAGE.CROP_HEIGHT), :, :]

        # resize image
        img_resized = cv2.resize(img_cropped, dsize=(w, h))

        # eventually change color space
        if self.cfg.MODEL.CNN.INPUT_CHANNELS == 1:
            img_resized = np.expand_dims(cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV)[:, :, 0], 2)

        return img_resized.astype('float32')


def worker(image, path):
    image = image.permute(0, 3, 1, 2)
    save_image(image, path + '_image.jpg', normalize=True, padding=0)


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def do_visualization(
        cfg,
        model,
        device,
):
    loss_list=[]
    err_list = []
    dataset = DrivingDatasetDataset(cfg)
    dataset.augment_data = False

    # Make data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

    # logger = setup_logger("", cfg.OUTPUT.DIR,'')

    create_path(cfg.OUTPUT.VIS_DIR)

    for iteration, (images, targets) in tqdm(enumerate(dataloader)):
        images = images.to(device)
        targets = targets.to(device)
        predictions, loss = model(images, targets)
        # image = images[0, :, :, :]
        # path = cfg.OUTPUT.VIS_DIR + "/" + str(iteration)
        # worker(image, path)
        floss = loss.cpu().item()
        loss_list.append(floss)
        if floss > 1000:
            err_list.append(iteration)
        # logger.info(loss.cpu().item())
    return loss_list,err_list

def visualization(cfg):
    # build the model
    model = build_model(cfg)

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model.eval()

    # load last checkpoint
    model.load_state_dict(torch.load("checkpoints\\weights_final.pth"))

    # build the dataloader

    # start the visualization procedure
    loss_list, err_list = do_visualization(
        cfg,
        model,
        device,
    )
    x = np.arange(len(loss_list))
    y = np.array(loss_list)
    plt.xlabel("frame")
    plt.ylabel("loss")
    plt.plot(x, y)
    plt.savefig('err.jpg')

    plt.show()
    with open("err.txt",'w') as f:
        for err in err_list:
            f.write(str(err)+"\n")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Self-driving Car Training and Inference.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--mode",
        default="test",
        metavar="mode",
        help="'train' or 'test'",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    visualization(cfg)


if __name__ == "__main__":
    main()
