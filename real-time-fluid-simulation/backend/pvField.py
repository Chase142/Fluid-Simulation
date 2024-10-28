import torch

class PvField:
    
    velo : tuple[torch.tensor, torch.tensor] = None
    pressure : torch.tensor = None
    
    
    def __init__(self, v : tuple[torch.tensor, torch.tensor], p : torch.tensor):
        self.velo = v
        self.pressure = p