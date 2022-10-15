import torch

data = [[659.2447, 194.8682, 780.5992, 305.2734]]
bboxes1 = torch.tensor(data)

print(bboxes1[..., :, None, :2])
print(bboxes1[..., None, :, :2])
print(bboxes1[..., :, None, 2:])
print(bboxes1[..., None, :, 2:])
