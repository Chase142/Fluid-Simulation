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
tau = 0.6
u0 = 0.125

omega1 = dt/tau
omega2 = 1-omega1
omega = np.array([omega1, omega2])

STEPS_PER_UPDATE = 100

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


obj_type = 'circle'
solid, boundary_nodes = define_object(obj_type, c, q, Nx, Ny, X, Y)
        

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

            vSq = u[0]**2 + u[1]**2
            
            vSq = vSq / np.max(vSq)

            rhoNorm = rho - np.min(rho)
            rhoNorm = rho / np.max(rho)
            
            rhoNorm = np.where(solid == 1, 0, rhoNorm)

            payload = [vSq.tolist(), rhoNorm.tolist()]

            socketio.emit('simulation_update', payload, room=sid)

            eventlet.sleep(1 / 60)

        print(f"Simulation task for SID {sid} has been terminated.")

    simulation_greenlet = socketio.start_background_task(simulation_task, sid, stop_event)

    connections[sid] = {
        'greenlet': simulation_greenlet,
        'stop_event': stop_event,
    }

@socketio.on('param_update')
def handle_update(params):
    global u0, omega, u, f, solid, boundary_nodes, obj_type
    u0 = params["inletVelocity"]
    tau = params["tau"]
    obj_shape = params["shape"]
    obj_type = obj_shape
    solid, boundary_nodes = define_object(obj_shape, c, q, Nx, Ny, X, Y)
    omega1 = dt/tau
    omega2 = 1-omega1
    omega = np.array([omega1, omega2])

@socketio.on('change_res')
def handle_res(params):
    global Nx, Ny, N
    Nx = params["nx"]
    Ny = params["ny"]
    N = Nx * Ny
    handle_reset()

@socketio.on('stop_simulation')
def handle_stop():
    sid = request.sid
    if sid in connections:
        connections[sid]['stop_event'].send('stop')
        del connections[sid]

@socketio.on('reset_simulation')
def handle_reset():
    global f, f_eq, f_star, u, rho, solid, boundary_nodes, indexes, x, X, y, Y
    x = np.arange(Nx)+0.5
    y = np.arange(Ny)+0.5
    X, Y = np.meshgrid(x,y)
    f = np.zeros((q, Ny, Nx), dtype=float_type)
    f_eq = np.zeros((q, Ny, Nx), dtype=float_type)
    f_star = np.zeros((q, Ny, Nx), dtype=float_type)
    rho = np.ones((Ny, Nx), dtype=float_type)
    u = np.zeros((d, Ny, Nx), dtype=float_type)
    solid, boundary_nodes = define_object(obj_type, c, q, Nx, Ny, X, Y)

    indexes = np.zeros((q, Nx * Ny), dtype=int)
    for i in range(q):
        xArr = (np.arange(Nx) - c[i][0] + Nx) % Nx
        yArr = (np.arange(Ny) - c[i][1] + Ny) % Ny

        xInd, yInd = np.meshgrid(xArr, yArr)

        indTotal = yInd * Nx + xInd
        indexes[i] = indTotal.reshape(Nx * Ny)

    u[0] += u0
    f = get_eq(f, rho, u, c, ceq, w, q)

    sid = request.sid
    if sid in connections:
        connections[sid]['stop_event'].send('stop')
        del connections[sid]

    handle_simulation()



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
