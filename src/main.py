import time
import cv2
import os
import datetime as dt
import cupy as cp
import numpy as np
from Tree_Algorythm import Cell
from Low_Count_Algorythms import Algorythms

def iteration_calc_np(bodies, G, DIMENSIONS):
    for i in range(bodies.shape[0]):
        for k in range(bodies.shape[0]):
            directon = bodies[k, :2]-bodies[i, :2]
            distance_qb = (directon[0]**2+directon[1]**2)**(3/2)
            if distance_qb < 8:
                continue
            bodies[i, 2:] += G/distance_qb*directon
        bodies[i, :2] += bodies[i, 2:]
        print(f"b: {i}")
    return bodies

def TreeAlgorythm(bodies, G):
    Tree = Cell(bodies, [np.amin(bodies[:, 0]), np.amin(bodies[:, 1])], [np.amax(bodies[:, 0])-np.amin(bodies[:, 0], np.amax[bodies[:, 1]-np.amin(bodies[:, 1])])], G)
    # Tree = 
    return bodies
    
def render(bodies, out, BORDER_COLOR, RESOLUTION):
            
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


G = 1e1
N = 10000
DIMENSIONS = 2
RESOLUTION = [1000, 1000]
ITERATIONS = 1000
FPS = 25
DURATION = 5
VIDEO_TIME = 60
ITERATIONS_PER_FRAME = 10
BORDER_COLOR = 100

bodies = (np.random.rand(N, DIMENSIONS*2) * np.concatenate((RESOLUTION, np.array([0, 0])), axis=0)).astype("float")

out = cv2.VideoWriter("./videos/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (RESOLUTION[1], RESOLUTION[0]), False)
Algs = Algorythms()


frame_time = 0
START_TIME = time.time()
iteration = 0

while iteration/FPS < VIDEO_TIME:
# while time.time() - START_TIME < DURATION:
# for iteration in range(int(ITERATIONS)):
    iteration += 1

    # bodies = iteration_calc_np(bodies, G, DIMENSIONS)
    # bodies = matricies_calc_np(bodies, G, DIMENSIONS)
    bodies = Algs.matricies_calc_cp(bodies, G, DIMENSIONS)

    if iteration % ITERATIONS_PER_FRAME == 0:
    # if (time.time() - frame_time) > 1/FPS:

        os.system("clear")
        print("Seconds: ", dt.timedelta(seconds=int(time.time() - START_TIME)))
        print("Iterations: ", iteration, "/", ITERATIONS)
        print("output time: ", iteration/FPS, "/", VIDEO_TIME)
        print("ETA: ", dt.timedelta(seconds=(time.time()-frame_time)*VIDEO_TIME*FPS/ITERATIONS_PER_FRAME-time.time()+START_TIME), "/", dt.timedelta(seconds=(time.time()-frame_time)*VIDEO_TIME*FPS/ITERATIONS_PER_FRAME))

        out = render(bodies, out, BORDER_COLOR, RESOLUTION)

        frame_time = time.time()
        if os.path.exists("./exit"):
            os.remove("./exit")
            break
    
out.release()
print(iteration)
# np.savez_compressed("./data/history", data)