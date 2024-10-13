from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''
class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):       # AT模型初始化 p为几次方
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):   # 前向传播
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))  #教师模型和学生模型的军方损失

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)       # 按行相加，
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am