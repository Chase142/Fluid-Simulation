import eventlet
eventlet.monkey_patch()
import numpy as np
from simulation import *
from eventlet.event import Event
from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet', ping_timeout=120, ping_interval=25, logger=False, engineio_logger=False)

float_type = np.float64

# D2Q9 lattice
d = 2
q = 9

Nx = 200
Ny = 80
N = Nx * Ny

# Lattice Units
dx = 1.
dt = 1.
rho = 1.

# Lattice Coordinates
x = np.arange(Nx)+0.5
y = np.arange(Ny)+0.5
X, Y = np.meshgrid(x,y)

# Paramters
tau = 0.545
u0 = 0.125

omega1 = dt/tau
omega2 = 1-omega1
omega = np.array([omega1, omega2])

STEPS_PER_UPDATE = 1

# Lattice velocities & speed of sound
c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1, 1], [-1,-1], [1,-1]])
cT = c.T
cs = 1/np.sqrt(3)

directions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
opposite_directions = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Lattice Weights
w0 = 4./9
w1 = 1./9
w2 = 1./36
w = np.array([w0, w1, w1, w1, w1, w2, w2, w2, w2])

# Equilibrium constants
c1 = 1.0
c2 = 3.0
c3 = 9./2
c4 = -3./2

ceq = np.array([c1, c2, c3, c4])

# Initializing arrays
f = np.zeros((q, Ny, Nx), dtype=float_type)
f_eq = np.zeros((q, Ny, Nx), dtype=float_type)
f_star = np.zeros((q, Ny, Nx), dtype=float_type)
rho = np.ones((Ny, Nx), dtype=float_type)
u = np.zeros((d, Ny, Nx), dtype=float_type)

# Pre-calculating streaming indexes
indexes = np.zeros((q, Nx * Ny), dtype=int)
for i in range(q):
    xArr = (np.arange(Nx) - c[i][0] + Nx) % Nx
    yArr = (np.arange(Ny) - c[i][1] + Ny) % Ny

    xInd, yInd = np.meshgrid(xArr, yArr)

    indTotal = yInd * Nx + xInd
    indexes[i] = indTotal.reshape(Nx * Ny)


# Pre-calculating solid & solid bounce backs
solid = np.zeros((Ny, Nx), dtype=bool)
boundary_nodes = np.zeros((q, Ny, Nx), dtype=bool)

# Square
# solid = np.where((X > (Nx // 12)) & (X < ((Nx*2)// 12)) & (Y > (Ny // 3)) & (Y < ((Ny*2)// 3)), True, False)

# Circle
circle_center_x = Nx // 6
circle_center_y = Ny // 2
circle_radius = Ny // 6
distances = np.sqrt((X- circle_center_x)**2 + (Y-circle_center_y)**2)
solid = np.where((distances < circle_radius), True, False)

# Line
# wall_height = 8
# solid[:, Ny//2] = np.where((Y[:, Ny//2]<int((Ny//2)+wall_height)) & (Y[:, Ny//2]>int((Ny//2)-wall_height)), True, False)

for i in range(q):
    streamed = np.roll(solid, shift=-c[i, 0], axis=1)
    streamed = np.roll(streamed, shift=-c[i, 1], axis=0)
    boundary_nodes[i] = (streamed==True) & (solid==False)


# Initialize distributions (steady flow right)
u[0] += u0
f = get_eq(f, rho, u, c, ceq, w, q)

connections = {}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_simulation')
def handle_simulation():
    sid = request.sid 

    if sid in connections:
        connections[sid]['stop_event'].send('stop')
        print(f"Previous simulation for SID {sid} has been stopped.")

    stop_event = Event()

    def simulation_task(sid, stop_event):
        global f, rho, u
        while True:
            if stop_event.ready():
                break

            f, rho, u = update(f, f_eq, f_star, rho, u, u0, c, cT, ceq, w, q, N, Ny, Nx, omega, indexes, boundary_nodes, opposite_directions, STEPS_PER_UPDATE)

            payload = u.tolist()

            socketio.emit('simulation_update', payload, room=sid)

            eventlet.sleep(1 / 60)

        print(f"Simulation task for SID {sid} has been terminated.")

    simulation_greenlet = socketio.start_background_task(simulation_task, sid, stop_event)

    connections[sid] = {
        'greenlet': simulation_greenlet,
        'stop_event': stop_event,
    }

@socketio.on('stop_simulation')
def handle_stop():
    sid = request.sid
    if sid in connections:
        connections[sid]['stop_event'].send('stop')
        global total_time
        total_time = 0
        del connections[sid]

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in connections:
        connections[sid]['stop_event'].send('stop')
        global total_time
        total_time = 0
        print(f"Simulation for SID {sid} has been stopped.")
        del connections[sid]
    else:
        print(f"No simulation found for SID {sid} on disconnect.")

if __name__ == '__main__':
    socketio.run(app, debug=True)
