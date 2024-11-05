import eventlet
eventlet.monkey_patch() 

from simulation import Fluid
from eventlet.event import Event
import base64
import torch
from flask import Flask, render_template, request
from flask_socketio import SocketIO


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet', ping_timeout=120, ping_interval=25)

Nx = Ny = 100
U = 1
L = 1
T = L/U
u_inlet = 0.5
rho = 1
nu = 0.001
g = 9.8
max_iters = 100
tol = 1e-6
alpha = 0.1

x_p = 0.5
y_p = 0.5
r = 0.1

total_time = 0

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
        fluid = Fluid(Nx, Ny, U, L, rho, nu, u_inlet, g=g, max_iters=max_iters, tol=tol, alpha=alpha)
        fluid.s, fluid.s_obj, fluid.s_obj_boundary, fluid.s_left, fluid.s_right, fluid.s_up, fluid.s_down = fluid.define_obj(x_p, y_p, r)
        fluid.u, fluid.v, fluid.p = fluid.apply_boundary_conditions()

        while True:
            if stop_event.ready():
                break
            p_byte = fluid.p.cpu().numpy().tobytes()
            u_byte = fluid.u.cpu().numpy().tobytes()
            v_byte = fluid.v.cpu().numpy().tobytes()

            k = fluid.u**2 + fluid.v**2
            # print('k', torch.sum(torch.sqrt(k)))

            p64 = base64.b64encode(p_byte).decode('utf-8')
            u64 = base64.b64encode(u_byte).decode('utf-8')
            v64 = base64.b64encode(v_byte).decode('utf-8')

            shape = fluid.p.shape
            dtype = fluid.p.dtype

            metadata = {
                'shape': shape,
                'dtype': str(dtype)
            }

            payload = {
                'metadata': metadata,
                'p': p64,
                'u': u64,
                'v': v64
            }

            socketio.emit('simulation_update', payload, room=sid)

            _, _, _, dt = fluid.update()

            global total_time
            # total_time += dt
            # print(total_time*T)
            eventlet.sleep(1/120)

        print(f"Simulation task for SID {sid} has been terminated.")


    simulation_greenlet = socketio.start_background_task(simulation_task, sid, stop_event)

    connections[sid] = {
        'greenlet': simulation_greenlet,
        'stop_event': stop_event
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

@socketio.on('mouse_move')
def handle_mouse_move(position):
    print(f"data received")

if __name__ == '__main__':
    socketio.run(app, debug=True)
