import torch
from pose_model import PoseModelForConversion

model = PoseModelForConversion()
model.load_state_dict(torch.load("saved_models/yaw_network.pth"))
#model.eval()
script_model = torch.jit.script(model)
script_model.save("saved_models/pose.pt")
print("Done")
