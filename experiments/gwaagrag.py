# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:58:36 2026

@author: mikun
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


a = 10.0                 
b = 6.0                    
epsilon = 0.2

x1 = 2.0                  
y1 = 2.5                 
vx1 = 2.8                
vy1 = 1.5               

x2 = 7.5               
y2 = 3.8              
vx2 = -2.1              
vy2 = -1.9             

dt = 0.01       
total_time = 60.0        
trail_length = 300        

n_steps = int(total_time / dt)
positions1 = np.zeros((n_steps, 2))
positions2 = np.zeros((n_steps, 2))

x = np.array([x1, x2])
y = np.array([y1, y2])
vx = np.array([vx1, vx2])
vy = np.array([vy1, vy2])

positions1[0] = [x[0], y[0]]
positions2[0] = [x[1], y[1]]

for i in range(1, n_steps):
    x += vx * dt
    y += vy * dt

    # Wall bounces (perfect reflection)
    for j in range(2):
        if x[j] <= 0:
            x[j] = 0
            vx[j] = -vx[j]
        elif x[j] >= a:
            x[j] = a
            vx[j] = -vx[j]

        if y[j] <= 0:
            y[j] = 0
            vy[j] = -vy[j]
        elif y[j] >= b:
            y[j] = b
            vy[j] = -vy[j]

    dx = x[0] - x[1]
    dy = y[0] - y[1]
    rel_vel_dot = dx * (vx[0] - vx[1]) + dy * (vy[0] - vy[1])
    if abs(dx) <= epsilon and abs(dy) <= epsilon and rel_vel_dot < 0:
        vx[0], vx[1] = vx[1], vx[0]
        vy[0], vy[1] = vy[1], vy[0]

    positions1[i] = [x[0], y[0]]
    positions2[i] = [x[1], y[1]]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, a)
ax.set_ylim(0, b)
ax.set_aspect('equal')
ax.set_title('Billiard Table Simulation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

rect = plt.Rectangle((0, 0), a, b, fill=False, linewidth=4, edgecolor='black')
ax.add_patch(rect)

ball1, = ax.plot([], [], 'o', markersize=14, color='red', label='Ball 1')
ball2, = ax.plot([], [], 'o', markersize=14, color='blue', label='Ball 2')
trail1, = ax.plot([], [], '-', color='red', alpha=0.35, lw=2)
trail2, = ax.plot([], [], '-', color='blue', alpha=0.35, lw=2)

def init():
    ball1.set_data([], [])
    ball2.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    return ball1, ball2, trail1, trail2

def animate(frame):
    # Current ball positions
    ball1.set_data([positions1[frame, 0]], [positions1[frame, 1]])
    ball2.set_data([positions2[frame, 0]], [positions2[frame, 1]])

    # Trails (last trail_length points)
    start = max(0, frame - trail_length)
    trail1.set_data(positions1[start:frame+1, 0], positions1[start:frame+1, 1])
    trail2.set_data(positions2[start:frame+1, 0], positions2[start:frame+1, 1])

    return ball1, ball2, trail1, trail2

ani = FuncAnimation(fig, animate, frames=n_steps, init_func=init,
                    interval=dt*1000*0.6, blit=True, repeat=True)

plt.legend(loc='upper right')
plt.show()