import torch
import torch.nn as nn
import torch.nn.functional as F 

def gram_matrix(input):
	a, b, c, d = input.size()
	# a = batch size (=1)
	# b = number of feature map
	# c,d = dimenstion of a feature map (N=c*d)
	
	features = input.view(a * b, c * d)
	G = torch.mm(features, features.t())
	return G.div(a * b * c * d)


class StyleLoss(nn.Module): 
	
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = gram_matrix(target_feature).detach()
		
	def forward(self, input):
		G = gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input