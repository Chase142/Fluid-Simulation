const socket = io.connect('http://' + document.domain + ':' + location.port);

const canvas = document.getElementById('drawingCanvas');
// Initialize the GL context
const gl =
    canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
// Only continue if WebGL is available and working
if (gl === null) {
  alert(
    "Unable to initialize WebGL. Your browser or machine may not support it.",
  );
}

// const ctx = canvas.getContext('2d');
let drawing = false;
var xpoints = [];
var ypoints = [];

var heatFragCode = document.getElementById("heatFrag").textContent;
var heatVertCode = document.getElementById("heatVert").textContent;
const heatmap = new Heatmap(canvas, 80, 200, heatVertCode, heatFragCode);
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
        xpoints.push(x);
        ypoints.push(y);
    }
});

document.addEventListener('mouseup', () => {
    drawing = false;
});

var brushRadius = 5;

document.getElementById('inletVelocity').addEventListener('change', () => {
    
    socket.emit('param_update', 
        {
            inletVelocity: parseFloat(document.getElementById('inletVelocity').value),
            tau: parseFloat(document.getElementById('viscosity').value)
        });
});

document.getElementById('viscosity').addEventListener('change', () => {
    
    socket.emit('param_update', 
        {
            inletVelocity: parseFloat(document.getElementById('inletVelocity').value),
            tau: parseFloat(document.getElementById('viscosity').value)
        });
});

document.getElementById('startSimulation').addEventListener('click', () => {
    console.log(xpoints)
    console.log(ypoints)
    const params = {
        inletVelocity: parseFloat(document.getElementById('inletVelocity').value),
        brushPoints: [xpoints, ypoints],
        brushRadius: brushRadius
    };
    socket.emit('start_simulation', params );
    // socket.emit('start_simulation');
});

socket.on('simulation_update', function(payload){
    decodeSimulationData(payload);
});