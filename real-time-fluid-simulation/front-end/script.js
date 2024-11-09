// Initialize Socket.IO connection
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
});

document.getElementById('startSimulation').addEventListener('click', () => {
    if (points.length > 0) {
        const params = {
            inletVelocity: parseFloat(document.getElementById('param1').value),
        };
        socket.emit('start_simulation', { points, params });
        alert('Simulation started!');
    } else {
        alert('Please draw a wing shape before starting the simulation.');
    }
});

// Handle incoming simulation data
socket.on('simulation_update', (data) => {
    decodeSimulationData(data);
});

function decodeSimulationData(data) {
    const { p, u, v, metadata } = data;
    const { shape, dtype } = metadata;

    const pArrayBuffer = Uint8Array.from(atob(p), c => c.charCodeAt(0)).buffer;
    const uArrayBuffer = Uint8Array.from(atob(u), c => c.charCodeAt(0)).buffer;
    const vArrayBuffer = Uint8Array.from(atob(v), c => c.charCodeAt(0)).buffer;

    const pData = new Float32Array(pArrayBuffer);
    const uData = new Float32Array(uArrayBuffer);
    const vData = new Float32Array(vArrayBuffer);

    visualizeSimulation(pData, uData, vData, shape);
}

function visualizeSimulation(pData, uData, vData, shape) {
    const width = shape[1];
    const height = shape[0];
    
    const imgData = ctx.createImageData(width, height);

    for (let i = 0; i < pData.length; i++) {
        const value = pData[i];
        const color = Math.floor((value + 1) * 127.5); // Normalize for color

        imgData.data[i * 4] = color;
        imgData.data[i * 4 + 1] = color;
        imgData.data[i * 4 + 2] = color;
        imgData.data[i * 4 + 3] = 255; // Alpha channel
    }

    ctx.putImageData(imgData, 0, 0);
}
