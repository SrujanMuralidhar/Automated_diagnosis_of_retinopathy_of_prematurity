import torch

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Device: " ,device)

# If num_classes == 2
'''
0 -> No RoP
1 -> RoP

'''

# If num_classes == 3
'''
0 -> Stage 1
1 -> Stage 2
2 -> Stage 3

'''
ROP_MAP = {0: "No RoP",1 : "RoP"}
STAGE_WISE_MAP = {0 : "Stage 1", 1: "Stage 2",2: "Stage 3"}
size = (512, 512)
