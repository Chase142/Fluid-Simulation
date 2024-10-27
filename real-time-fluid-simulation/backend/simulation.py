import torch
import numpy as np

import fluid
import pvField

class Simulation:
    
    #Facts
    g = 9.8 # gravitational constant [m/s^2]
    
    #Materials
    defaultFluid : fluid.Fluid = None
    field : pvField.PvField = None
    
    #Geometry
    stepSize : np.ndarray = (1, 1)
    dtMin : float = 1
    cellSize : np.ndarray = (1,1)
    nCell : np.ndarray = (1,1)
    
    #Fake Numbers
    charVelo : float = 1
    charLen  : float = 1
    charTime : float = 1
    charPressure : float = 1
    defReynolds : float = 1
    defFroude   : float = 1
    
    #Util
    torchDevice : torch.device = torch.cpu
    
    def __init__(self, defFluid : fluid.Fluid, pvInitial : pvField.PvField,
                 dtMin : float, nCell : np.ndarray, cell : np.ndarray,
                 charVelo : float,  torchDevice : torch.device = torch.cpu):
        self.defaultFluid = defFluid
        self.field = pvInitial
        
        self.dtMin = dtMin
        self.cellSize = cell
        self.nCell = nCell
        
        self.charVelo = charVelo
        
        self.torchDevice = torchDevice
        
    @classmethod
    def _initializeCell(self):
        self.stepSize = (self.cellSize / self.nCell) / self.charVelo
        
    @classmethod
    def _characterize(self):
        self.charTime = self.charLen / self.charVelo # Characteristic time [s]
        self.charPressure = self.defaultFluid.density * (self.charVelo**2) # Characteristic pressure [Pa]

        self.defReynolds = self.charVelo *  self.charLen / self.defaultFluid.viscosity # Reynolds number
        self.defFroude = self.charVelo / ((self.g * self.charLen)**(1/2)) # Froude number
        