from model import Discriminator
import torch
import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import argparse

transform_input = A.Compose([
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
def get_model():
    discriminator = Discriminator().cuda()
    model_file_path = "./weights/IFQA_Metric.pth"
    discriminator.load_state_dict(torch.load(model_file_path)['D'])
    discriminator.eval()
    return discriminator
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform_input(image=img)['image'].unsqueeze(0).cuda()
    return img

def evaluate_img(img_path, model):
    img = read_image(img_path)
    with torch.no_grad():
        p_map = model(img)
        sum = torch.sum(p_map) / (3*256*256)
        score = round(torch.mean(sum).detach().cpu().item(), 4)
    p_map = p_map.squeeze().cpu().numpy()
    p_map = p_map[..., np.newaxis]
    p_map *= 255
    p_map = p_map.astype(np.uint8)
    c_map = cv2.applyColorMap(p_map, colormap=16)
    saved_name = img_path.split("/")[-1]
    cv2.imwrite("./results/{0}".format(saved_name), c_map)
    print("File: {0} Score: {1}".format(saved_name, score))


def main(config):
    discriminator = get_model()
    f_path = config.f_path
    print("Start assessment..")
    if os.path.isfile(f_path):
        evaluate_img(f_path, discriminator)
    else:
        files_list = os.walk(f_path).__next__()[2]
        for file_name in files_list:
            file_path = os.path.join(f_path, file_name)
            evaluate_img(file_path, discriminator)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_path', type=str, default="./docs/", help='file path or folder path')
    config = parser.parse_args()
    print(config)
    main(config)