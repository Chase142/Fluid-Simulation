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
let points = [];

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
document.addEventListener('mousedown', () => {
    drawing = true;
    points = [];
});

canvas.addEventListener('mousemove', (event) => {
    if (drawing) {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / canvas.width;
        const y = (event.clientY - rect.top) / canvas.height;
        points.push({ x, y });
        console.log(points);
    }
});

document.addEventListener('mouseup', () => {
    drawing = false;
});

document.getElementById('clearCanvas').addEventListener('click', () => {
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    points = [];
    alert('Cleared canvas not implemented yet');
});

document.getElementById('clearBarriers').addEventListener('click', () => {
    socket.emit('clear_barriers');
    alert('Cleared barries not immplemented yet');
});

document.getElementById('startSimulation').addEventListener('click', () => {
    // const params = {
    //     inletVelocity: parseFloat(document.getElementById('param1').value)
    // };
    // socket.emit('start_simulation', { points, params });
    socket.emit('start_simulation');
});

socket.on('simulation_update', function(payload){
    decodeSimulationData(payload);
});

document.getElementById('resetFluid').addEventListener('click', () => {
    socket.emit('reset_fluid');
    alert('Reset not implemented yet');
});
