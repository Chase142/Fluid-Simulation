from flask import Flask, render_template
from flask_socketio import SocketIO
import eventlet

# Initialize Flask app and Flask-SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

@app.route('/')
def index():
    """Serve the index HTML file."""
    return render_template('index.html')

# WebSocket event when the frontend connects and requests simulation data
@socketio.on('start_simulation')
def handle_simulation():
    """Send test data to the client every second."""
    while True:
        test_data = {'message': 'Hello from Flask!'}
        socketio.emit('simulation_update', test_data)  # Emit the data to the frontend
        socketio.sleep(1)  # Send data every 1 second

if __name__ == '__main__':
    socketio.run(app, debug=True)
