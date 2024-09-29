import torch.nn.functional as F
import torch
inp1=torch.rand(10,3).log()
inp2=torch.rand(10,3)
print(torch.min(inp1),torch.max(inp1),'inp1')
print(torch.min(inp2),torch.max(inp2),'inp2')
kl_criterion=torch.nn.KLDivLoss()
print(kl_criterion(inp1,inp2))

input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
target = F.softmax(torch.rand(3, 5), dim=1)
output = kl_criterion(input, target)
print(target,'target')
print(input,'input')
print(output,'output')