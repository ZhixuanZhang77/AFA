"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import ToTensor, ToPILImage, transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
import ssl
from torchvision.models import *

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of sampled examples")
parser.add_argument("--delta", type=float, default=0.5, help="The balanced coefficient")
parser.add_argument("--zeta", type=float, default=3.0, help="The upper bound of neighborhood")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

net_name = 'incv3'# 'incv3','incv4','res50','res101,'ViT-B/16','Swin-T'

if net_name in ['ViT-B/16', 'PiT-S', 'MLP-mixer', 'ResMLP', 'Swin-T']:
    img_size = 224
else:
    img_size = 299

transforms = T.Compose(
    [T.Resize(img_size), T.ToTensor()]
)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

def AFA(images, gt, model, min, max):  # success 100 95.2 72.80 73.00
    """
    The attack algorithm of our proposed CMI-FGSM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation
    :param max: the max the clip operation
    :return: the adversarial images
    """
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()
    zeta = opt.zeta
    delta = opt.delta
    N = opt.N
    grad = torch.zeros_like(x).detach().cuda()

    b = alpha * delta
    m = 0.9

    gamma = 0.15 * eps

    for i in range(num_iter):
        avg_grad = torch.zeros_like(x).detach().cuda()
        momentum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x + torch.rand_like(x).uniform_(-eps * zeta, eps * zeta)

            sample_min = clip_by_tensor(x_near - gamma, 0.0, 1.0)
            sample_max = clip_by_tensor(x_near + gamma, 0.0, 1.0)
            x_near = x_near - gamma * torch.sign(momentum_grad)
            x_near = clip_by_tensor(x_near, sample_min, sample_max)

            x_t = V(x_near.detach(), requires_grad=True)
            g_t0 = torch.autograd.grad(F.cross_entropy(model(x_t), gt), x_t, retain_graph=False, create_graph=False)[0]
            x_t1 = x_t.detach() - g_t0 * alpha / (torch.abs(g_t0).mean([1, 2, 3], keepdim=True) + 1e-12)

            x_t1 = V(x_t1.detach(), requires_grad=True)
            g_t1 = torch.autograd.grad(F.cross_entropy(model(x_t1), gt), x_t1, retain_graph=False, create_graph=False)[
                0]
            h_t0 = g_t1 - g_t0

            x_t2 = x_t.detach() - h_t0.detach() * alpha / (
                        torch.abs(h_t0.detach()).mean([1, 2, 3], keepdim=True) + 1e-12)  #
            x_t2 = V(x_t2.detach(), requires_grad=True)
            g_t2 = torch.autograd.grad(F.cross_entropy(model(x_t2), gt), x_t2, retain_graph=False, create_graph=False)[
                0]
            x_t3 = x_t2.detach() - g_t2 * alpha / (torch.abs(g_t2).mean([1, 2, 3], keepdim=True) + 1e-12)

            x_t3 = V(x_t3.detach(), requires_grad=True)
            g_t3 = torch.autograd.grad(F.cross_entropy(model(x_t3), gt), x_t3, retain_graph=False, create_graph=False)[
                0]
            h_t1 = g_t3 - g_t2

            grad_ = g_t0 + g_t1 + g_t2 + g_t3 + b * (delta * h_t0 + (1 - delta) * h_t1)  
            avg_grad += grad_

            momentum_grad = m * momentum_grad - g_t0
        noise = avg_grad / (torch.abs(avg_grad).mean([1, 2, 3], keepdim=True) + 1e-12)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def main():

    if net_name == 'incv3':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'incv4':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'res50':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'res101':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'ViT-B/16':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    vit_b_16(num_classes=1000, pretrained=True).eval().cuda())  # Normalization unknown
    elif net_name == 'Swin-T':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    swin_t(num_classes=1000, pretrained=True).eval().cuda())

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img = AFA(images, gt, model, images_min, images_max)

        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, './{}_AFA_outputs/'.format(net_name))


if __name__ == '__main__':
    main()
