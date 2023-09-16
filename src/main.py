from matplotlib import pyplot as plt
import time
import numpy as np
import cv2
import os
import cupy as cp
# 


def matricies_calc_np(bodies, G, DIMENSIONS):
    G_VECTOR = np.array((G, G))
    M = np.expand_dims(bodies[:, :2], axis=0).astype("short")
    E = (np.transpose(M, (1, 0, 2)) - M)
    bodies[:, 2:] -= np.sum(np.divide(np.expand_dims(G_VECTOR, axis=(0, 1)), E, out=np.zeros((N, N, DIMENSIONS)), where=E!=0), axis=1)
    bodies[:, :2] += bodies[:, 2:]
    return bodies
    
def render(bodies, out, BORDER_COLOR, RESOLUTION):
        # os.system("clear")
        # print("Seconds: ", time.time() - START_TIME)
        frame_time = time.time()
    
        frame = np.zeros(RESOLUTION, dtype="uint8")
        frame[:, 0] = BORDER_COLOR
        frame[:, RESOLUTION[1]-1] = BORDER_COLOR
        frame[0, :] = BORDER_COLOR
        frame[RESOLUTION[0]-1, :] = BORDER_COLOR

        # mask = bodies     
        for i in range(bodies.shape[0]):
            if bodies[i, 0] < RESOLUTION[0] and bodies[i, 0] > 0 and bodies[i, 1] < RESOLUTION[1] and bodies[i, 1] > 0:
                frame[int(bodies[i, 0]), int(bodies[i, 1])] += 100 
    
        out.write(frame)
        return out




G = 1e-2
RESOLUTION = np.array([720, 1280])
N = 1000
DIMENSIONS = 2
ITERATIONS = 1e3
FPS = 25
DURATION = 5*1
SIMULATING_TIME = 10
BORDER_COLOR = 100
# size = 720*16//9, 720

bodies = (np.random.rand(N, DIMENSIONS*2) * np.concatenate((RESOLUTION, np.array([0, 0])), axis=0)).astype("float")
# data = np.ndarray((ITERATIONS, N, DIMENSIONS))


out = cv2.VideoWriter("./videos/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (RESOLUTION[1], RESOLUTION[0]), False)
frame_time = 0
START_TIME = time.time()
iteration = 0
while time.time() - START_TIME < DURATION:
# for iteration in range(int(ITERATIONS)):
    iteration += 1

    bodies = matricies_calc_np(bodies, G, DIMENSIONS)

    # if iteration % 100 == 0:
    if (time.time() - frame_time) > 1/FPS:
        out = render(bodies, out, BORDER_COLOR, RESOLUTION)
    
out.release()
print(iteration)
# np.savez_compressed("./data/history", data)