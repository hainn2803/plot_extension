import torch

at_checkpoint = "results/gradual_das_at_layer_selected.pt"
at_result = torch.load(at_checkpoint, map_location="cpu")
at_layer = int(at_result["layer"])
Q_at = at_result["basis"].float()

print(Q_at, at_layer)