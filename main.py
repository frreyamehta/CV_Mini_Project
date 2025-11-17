import torch

# adjust the path if you put it in a 'models/' folder
model_path = "LGC_best.pth"  

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # or 'cuda' if using GPU

# If it’s a full model
model = checkpoint  

# If it’s state_dict only
# model.load_state_dict(checkpoint)
