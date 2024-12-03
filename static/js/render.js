const socket = io.connect('http://' + document.domain + ':' + location.port);

const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;
let points = [];

// Drawing functionality
canvas.addEventListener('mousedown', () => {
    drawing = true;
    points = [];
    ctx.beginPath();
});

canvas.addEventListener('mousemove', (event) => {
    if (drawing) {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / canvas.width;
        const y = (event.clientY - rect.top) / canvas.height;
        points.push({ x, y });
        ctx.lineTo(event.clientX - rect.left, event.clientY - rect.top);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
});

document.getElementById('clearCanvas').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    points = [];
    alert('Cleared canvas not immplemented yet');
});

document.getElementById('clearBarriers').addEventListener('click', () => {
    socket.emit('clear_barriers');
    alert('Cleared barries not immplemented yet');
});

document.getElementById('startSimulation').addEventListener('click', () => {
    // const params = {
    //     inletVelocity: parseFloat(document.getElementById('param1').value),
    //     flowSpeed: parseFloat(document.getElementById('flowSpeed').value),
    //     viscosity: parseFloat(document.getElementById('viscosity').value),
    //     contrast: parseFloat(document.getElementById('contrast').value),
    //     plotType: document.getElementById('plotCurl').value,
    //     drawMode: document.getElementById('drawBarriers').value,
    //     barrierShape: document.getElementById('barrierShapes').value,
    //     tracers: document.getElementById('tracers').checked,
    //     flowlines: document.getElementById('flowlines').checked,
    //     forceOnBarriers: document.getElementById('forceOnBarriers').checked,
    //     sensor: document.getElementById('sensor').checked,
    //     data: document.getElementById('data').checked,
    // };
    // socket.emit('start_simulation', { points, params });
    socket.emit('start_simulation');
});

socket.on('simulation_update', function(payload) {
    console.log(payload)
});

document.getElementById('resetFluid').addEventListener('click', () => {
    socket.emit('reset_fluid');
    alert('Reset not implemented yet');
});
