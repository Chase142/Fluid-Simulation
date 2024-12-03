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




function decodeSimulationData(data) {
    // const { p, u, v, metadata } = data;
    // const { shape, dtype } = metadata;

    // const pArrayBuffer = Uint8Array.from(atob(p), c => c.charCodeAt(0)).buffer;
    // const uArrayBuffer = Uint8Array.from(atob(u), c => c.charCodeAt(0)).buffer;
    // const vArrayBuffer = Uint8Array.from(atob(v), c => c.charCodeAt(0)).buffer;

    // const pData = new Float32Array(pArrayBuffer);
    // const uData = new Float32Array(uArrayBuffer);
    // const vData = new Float32Array(vArrayBuffer);

    visualizeSimulation(data);
}


// Compile a WebGL program from a vertex shader and a fragment shader
compile = (gl, vshader, fshader) => {
    // Compile vertex shader
    var vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vshader);
    gl.compileShader(vs);
    
    // Compile fragment shader
    var fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fshader);
    gl.compileShader(fs);
    
    // Create and launch the WebGL program
    var program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.useProgram(program);
    
    // Log errors (optional)
    console.log('vertex shader:', gl.getShaderInfoLog(vs) || 'OK');
    console.log('fragment shader:', gl.getShaderInfoLog(fs) || 'OK');
    console.log('program:', gl.getProgramInfoLog(program) || 'OK');
    
    return program;
  }

var heatFragCode = document.getElementById("heatFrag").textContent;
var heatVertCode = document.getElementById("heatVert").textContent;

// Compile program
var program = compile(gl, heatVertCode, heatFragCode);

indexToCoord = (idx, size) => {
    return 2 * (idx / size) - 1;
}

function getMax(a){
    return Math.max(...a.map(e => Array.isArray(e) ? getMax(e) : e));
  }

function visualizeSimulation(data) {
    const width = data.length;
    const height = data[0].length;

    const posI = []

    for (let i = 0; i < width - 1; i++) {
        for (let j = 0; j < height - 1; j++){
            posI.push(indexToCoord(i + 1, width));
            posI.push(indexToCoord(j, height));
            posI.push(data[i + 1][j]);

            console.log(data[i][j])

            posI.push(indexToCoord(i, width));
            posI.push(indexToCoord(j, height));
            posI.push(data[i][j]);

            posI.push(indexToCoord(i, width));
            posI.push(indexToCoord(j + 1, height));
            posI.push(data[i][j + 1]);

            posI.push(indexToCoord(i, width));
            posI.push(indexToCoord(j + 1, height));
            posI.push(data[i][j + 1]);

            posI.push(indexToCoord(i + 1, width));
            posI.push(indexToCoord(j + 1, height));
            posI.push(data[i + 1][j + 1]);

            posI.push(indexToCoord(i + 1, width));
            posI.push(indexToCoord(j, height));
            posI.push(data[i + 1][j]);
        }
    }

    var posBuffer = new Float32Array(posI);

    // console.log(posBuffer[0]);
    // console.log(posBuffer[1]);
    // console.log(posBuffer[2]);

    // Get the size of each float in bytes (4)
    var FSIZE = posBuffer.BYTES_PER_ELEMENT;

    // Create a buffer object
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, posBuffer, gl.STREAM_DRAW);

    // Bind the attribute position to the 1st, 2nd
    var position = gl.getAttribLocation(program, 'position');
    gl.vertexAttribPointer(
      position,   // target
      2,          // interleaved data size
      gl.FLOAT,   // type
      false,      // normalize
      FSIZE * 3,  // stride (chunk size)
      0           // offset (position of interleaved data in chunk) 
    );
    gl.enableVertexAttribArray(position);

    // Bind the intensity to the 3rd element
    var intensity = gl.getAttribLocation(program, 'intensity');
    gl.vertexAttribPointer(
        intensity,      // target
      1,          // interleaved chunk size
      gl.FLOAT,   // type
      false,      // normalize
      FSIZE * 3,  // stride
      FSIZE * 2   // offset
    );
    gl.enableVertexAttribArray(intensity);

    // Set the clear color
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    // Clear canvas
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Render
    var numVerts = (width - 1) * (height - 1) * 6
    gl.drawArrays(gl.TRIANGLES, 0, numVerts);

    // ctx.putImageData(imgData, 0, 0);
}

// Drawing functionality
canvas.addEventListener('mousedown', () => {
    drawing = true;
    points = [];
    // ctx.beginPath();
});

canvas.addEventListener('mousemove', (event) => {
    if (drawing) {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / canvas.width;
        const y = (event.clientY - rect.top) / canvas.height;
        points.push({ x, y });
        // ctx.lineTo(event.clientX - rect.left, event.clientY - rect.top);
        // ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
});

document.getElementById('clearCanvas').addEventListener('click', () => {
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    gl
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

socket.on('simulation_update', function(payload){
    decodeSimulationData(payload);
});

document.getElementById('resetFluid').addEventListener('click', () => {
    socket.emit('reset_fluid');
    alert('Reset not implemented yet');
});
