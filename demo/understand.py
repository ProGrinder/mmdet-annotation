import torch

import torch.nn.functional as F
import torch.nn as nn
x = torch.randn(5, 5)
target = torch.tensor([0, 2, 3, 1, 4])
# 对标签进行one_hot编码
one_hot = F.one_hot(target).float()
softmax = torch.exp(x)/torch.sum(torch.exp(x), dim=1).reshape(-1, 1)
logsoftmax = torch.log(softmax)
nllloss = -torch.sum(one_hot*logsoftmax)/target.shape[0]
print(nllloss)
# 下面用torch.nn.function实现一下以验证上述结果的正确性
logsoftmax = F.log_softmax(x, dim=1)
# 无需对标签做one_hot编码
nllloss = F.nll_loss(logsoftmax, target)
print(nllloss)

# 最后我们直接用torch.nn.CrossEntropyLoss验证一下以上两种方法的正确性
cross_entropy = F.cross_entropy(x, target)

# data = [[659.2447, 194.8682, 780.5992, 305.2734]]
# bboxes1 = torch.tensor(data)
#
# print(bboxes1[..., :, None, :2])
# print(bboxes1[..., None, :, :2])
# print(bboxes1[..., :, None, 2:])
# print(bboxes1[..., None, :, 2:])
