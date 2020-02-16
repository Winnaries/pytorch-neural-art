from __future__ import print_function

import GradientDescent as gd

import torch
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 1100

loader = transforms.Compose([
	transforms.Resize(imsize),
	transforms.ToTensor()])

def image_loader(image_name):
	image = Image.open(image_name)
	image = loader(image).unsqueeze(0)
	return image.to(device, torch.float)

style_img = image_loader("./images/impressionist-tree.jpg")
content_img = image_loader("./images/tower.jpg")

assert style_img.size() == content_img.size(), \
	"we need to import style and content image of the same size"

unloader = transforms.ToPILImage()

plt.ion()

def imshow(tensor, title=None):
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	image = unloader(image)
	plt.imshow(image)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

plt.figure()
imshow(input_img, title='Input Image')

output = gd.run_style_transfer(device, cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, 200, 9000000)

plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()
