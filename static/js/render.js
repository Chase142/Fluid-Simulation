const socket = io.connect('http://' + document.domain + ':' + location.port);

const canvas = document.getElementById('drawingCanvas');
// Initialize the GL context
const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");

// Only continue if WebGL is available and working
if (gl === null) {
  alert(
    "Unable to initialize WebGL. Your browser or machine may not support it.",
  );
}

// const ctx = canvas.getContext('2d');
let drawing = false;
var points = [];

var heatFragCode = document.getElementById("heatFrag").textContent;
var heatVertCode = document.getElementById("heatVert").textContent;
const heatmap = new Heatmap(canvas, 
    parseInt(document.getElementById('ny').value), 
    parseInt(document.getElementById('nx').value), 
    heatVertCode, heatFragCode);
// Set the clear color
gl.clearColor(0.0, 0.0, 0.0, 1.0);

// Clear canvas
gl.clear(gl.COLOR_BUFFER_BIT);
heatmap.drawData();

function decodeSimulationData(data) {
    // Set the clear color
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    // Clear canvas
    gl.clear(gl.COLOR_BUFFER_BIT);
    heatmap.updateData(data[0], data[1]);
    heatmap.drawData();
    // console.log(data);
}

// Drawing functionality
document.addEventListener('clearCanvas', () => {
    drawing = false;
    console.log(points)
    points = [];
});

// Drawing functionality
canvas.addEventListener('mousedown', () => {
    drawing = true;
});

canvas.addEventListener('mousemove', (event) => {
    if (drawing) {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / canvas.width;
        const y = (event.clientY - rect.top) / canvas.height;
        points.push([x, y]);
    }
});

document.addEventListener('mouseup', () => {
    drawing = false;
});

var brushRadius = 5;

function updateParams(){
    viscosity = parseFloat(document.getElementById('viscosity').value);
    tau = 3*viscosity + 0.5;
    socket.emit('param_update', 
        {
            shape: document.getElementById('shape').value,
            inletVelocity: parseFloat(document.getElementById('inletVelocity').value),
            tau: tau,
        });
}

document.getElementById('inletVelocity').addEventListener('change', () => {
    updateParams();
});

document.getElementById('viscosity').addEventListener('change', () => {
    updateParams();
});

document.getElementById('shape').addEventListener('change', () => {
    updateParams();
});

document.getElementById('nx').addEventListener('change', () => {
    socket.emit('change_res', 
        {
            nx: parseInt(document.getElementById('nx').value),
            ny: parseInt(document.getElementById('ny').value),
        });
});

document.getElementById('ny').addEventListener('change', () => {
    socket.emit('change_res', 
        {
            nx: parseInt(document.getElementById('nx').value),
            ny: parseInt(document.getElementById('ny').value),
        });
});

document.getElementById('startSimulation').addEventListener('click', () => {
    socket.emit('start_simulation');
});

document.getElementById('stopSimulation').addEventListener('click', () => {
    socket.emit('stop_simulation');
});

document.getElementById('reset').addEventListener('click', () => {
    socket.emit('reset_simulation');
});

socket.on('simulation_update', function(payload){
    decodeSimulationData(payload);
});