from model import Discriminator
from model_huge import MaskedAutoencoderViT
import torch
import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn as nn
import argparse
transform_input = A.Compose([
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
transform_input2 = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
        ToTensorV2()
])
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
def read_image(img_path, model_name):
    if model_name == "VIT":
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform_input2(image=img)['image'].unsqueeze(0).cuda()
    elif model_name == "CNN":
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform_input(image=img)['image'].unsqueeze(0).cuda()
    return img

def evaluate_img(img_path, model, model_name):
    img = read_image(img_path, model_name)
    if model_name == "VIT":
        with torch.no_grad():
            real_latent, _, real_ids_restore = model.forward_encoder(img, mask_ratio=0)
            p_map = model.forward_decoder(real_latent, real_ids_restore)
            p_map = model.unpatchify(p_map, nc=1)
            zeros = torch.zeros(size=p_map.size()).cuda()
            preprocess = torch.where(p_map >= 0.000, p_map, zeros)
            sum = torch.sum(preprocess)
            sum_mean = sum / (224 * 224)
            score = round(torch.mean(sum_mean).detach().cpu().item(), 4)
    elif model_name == "CNN":
        with torch.no_grad():
            p_map = model(img)
            sum = torch.sum(p_map) / (256*256)
            score = round(torch.mean(sum).detach().cpu().item(), 4)
    p_map = p_map.squeeze().cpu().numpy()
    p_map = p_map[..., np.newaxis]
    p_map *= 255
    p_map = p_map.astype(np.uint8)
    c_map = cv2.applyColorMap(p_map, colormap=16)
    saved_name = img_path.split("/")[-1]
    cv2.imwrite("./results/{0}".format(saved_name), c_map)
    print("File: {0} Score: {1}".format(saved_name, score))
def get_model(model_name):
    if model_name == "VIT":
        model = MaskedAutoencoderViT()
        model.decoder_pred = nn.Linear(512, 14 ** 2 * 1, bias=True)
        model = model.cuda()
        model_file_path = "./weights/IFQA++_Metric.pth"
        model.load_state_dict(torch.load(model_file_path)['D'])
    elif model_name == "CNN":
        discriminator = Discriminator().cuda()
        model_file_path = "./weights/IFQA_Metric.pth"
        discriminator.load_state_dict(torch.load(model_file_path)['D'])
        discriminator.eval()
    return discriminator
def main(config):
    model_name = config.model
    discriminator = get_model(model_name)
    f_path = config.f_path
    print("Start assessment..")
    if os.path.isfile(f_path):
        evaluate_img(f_path, discriminator, model_name)
    else:
        files_list = os.walk(f_path).__next__()[2]
        for file_name in files_list:
            file_path = os.path.join(f_path, file_name)
            evaluate_img(file_path, discriminator, model_name)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_path', type=str, default="./docs/00021_Blurred.png", help='file path or folder path')
    parser.add_argument('--model', type=str, default="VIT", help='Choose two options: CNN or VIT')
    config = parser.parse_args()
    print(config)
    main(config)
