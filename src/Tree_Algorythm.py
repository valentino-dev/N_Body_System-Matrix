import numpy as np
from Low_Count_Algorythms import Algorythms

class Cell:
    THRESHHOLD = 4
    GRID_LENGTH = 2
    grid = {}
    Algs = Algorythms()

    def __init__(self, bodies: np.ndarray, position, size):

        for i in range(GRID_LENGTH):
            for k in range(GRID_LENGTH):
                mask = (bodies[:, 0] >= position+i*size/GRID_LENGTH and 
                bodies[:, 0] < position+(i+1)*size/GRID_LENGTH and 
                bodies[:, 1] >= position+k*size/GRID_LENGTH and 
                bodies[:, 1] < position+(k+1)*size/GRID_LENGTH)
                gird_bodies = bodies[mask]
                if gird_bodies.shape[0] > THRESHHOLD:
                    grid[i, k] = Cell(gird_bodies, [position+i*size/GRID_LENGTH, position+k*size/GRID_LENGTH], size/GRID_LENGTH)
                else:
                    pass
                    

        return self
                     
                

        







