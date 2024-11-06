const socket = io();
let selectedPlot = document.getElementById("plots").value;

window.onload = function() {
    socket.emit('start_simulation');
};

// window.onmousemove = function() {
//     socket.emit('mouse_move', window.onmousemove.arguments)
// }

document.getElementById("plots").addEventListener("change", function () {
    selectedPlot = this.value;
    plotInitialized = false
});

let plotData;
let plotLayout;
let plotInitialized = false;

let p;
let u;
let v;

socket.on('simulation_update', function(payload) {
    const metadata = payload.metadata;
    const p_dataBase64 = payload.p;
    const u_dataBase64 = payload.u
    const v_dataBase64 = payload.v

    if(selectedPlot == 'pressure')
    {
        const p_buffer = base64ToArrayBuffer(p_dataBase64);
        handleIncomingData_pressure(p_buffer, metadata);
    }
    if(selectedPlot == 'velocity')
    {
        const u_buffer = base64ToArrayBuffer(u_dataBase64);
        const v_buffer = base64ToArrayBuffer(v_dataBase64);
        handleIncomingData_velocity(u_buffer, v_buffer, metadata);
    }   
});

function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

function handleIncomingData_pressure(buffer, metadata) {
    p = reconstructArray(buffer, metadata);

    if (!plotInitialized) {
        initializePlot_pressure(p);
    } else {
        updatePlot_pressure(p);
    }
}

function handleIncomingData_velocity(u_buffer, v_buffer, metadata)
{
    u = reconstructArray(u_buffer, metadata)
    v = reconstructArray(v_buffer, metadata)

    if (!plotInitialized) {
        initializePlot_velocity(u, v);
    } else {
        updatePlot_velocity(u, v);
    }
}

function reconstructArray(buffer, metadata) {
    const { shape, dtype } = metadata;
    let typedArray;

    switch (dtype) {
        case 'torch.float32':
            typedArray = new Float32Array(buffer);
            break;
        case 'torch.float64':
            typedArray = new Float64Array(buffer);
            break;
        default:
            console.error('Unsupported data type:', dtype);
            return;
    }

    const array2D = [];
    for (let i = 0; i < shape[0]; i++) {
        array2D.push(Array.from(typedArray.slice(i * shape[1], (i + 1) * shape[1])));
    }

    return array2D;
}

 function initializePlot_pressure(data) {
    const trace = {
        z: data,
        type: 'heatmap',
        colorscale: 'Viridis',
    };

    plotData = [trace];

    plotLayout = {
        title: 'Tensor Data Heatmap',
        xaxis: {
            scaleanchor: 'y',
            scaleratio: 1,
            constrain: 'domain',
        },
        yaxis: {
            constrain: 'domain',
            autorange: 'reversed'
        },
        margin: {
            t: 50,
            r: 50,
            b: 50,
            l: 50,
        },
        autosize: true,
    };

    const config = { responsive: true };

    Plotly.newPlot('plotDiv', plotData, plotLayout, config);
    plotInitialized = true;
}


function updatePlot_pressure(data) {
    plotData[0].z = data;

    Plotly.react('plotDiv', plotData, plotLayout);
}


function initializePlot_velocity(u, v) {
    const x = Array.from({ length: u[0].length }, (_, i) => i);  // X grid
    const y = Array.from({ length: u.length }, (_, i) => i);      // Y grid

    const trace = {
        type: 'scattergl',
        mode: 'lines',
        line: { color: 'blue', width: 1 },
        x: [],
        y: [],
        opacity: 0.7,
    };

    // Generate streamline paths
    for (let i = 0; i < x.length; i += 5) {
        for (let j = 0; j < y.length; j += 5) {
            let xi = x[i];
            let yi = y[j];
            const pathX = [];
            const pathY = [];
            for (let k = 0; k < 20; k++) {  // Adjust step count for longer or shorter streamlines
                pathX.push(xi);
                pathY.push(yi);

                const u_val = u[Math.round(yi)][Math.round(xi)];
                const v_val = v[Math.round(yi)][Math.round(xi)];

                xi += u_val * 0.1;  // Adjust scaling factor for streamline length
                yi += v_val * 0.1;

                if (xi < 0 || xi >= x.length || yi < 0 || yi >= y.length) break;
            }
            trace.x.push(...pathX, null);  // Use null to separate each streamline
            trace.y.push(...pathY, null);
        }
    }

    plotData = [trace];

    plotLayout = {
        title: 'Streamline Plot',
        xaxis: {
            scaleanchor: 'y',
            scaleratio: 1,
            constrain: 'domain',
        },
        yaxis: {
            constrain: 'domain',
            autorange: 'reversed'
        },
        margin: {
            t: 50,
            r: 50,
            b: 50,
            l: 50,
        },
        autosize: true,
    };

    const config = { responsive: true };

    Plotly.newPlot('plotDiv', plotData, plotLayout, config);
    plotInitialized = true;
}


function updatePlot_velocity(u, v) {
    plotData[0].x = [];
    plotData[0].y = [];

    const x = Array.from({ length: u[0].length }, (_, i) => i);
    const y = Array.from({ length: u.length }, (_, i) => i);

    // Generate updated streamline paths
    for (let i = 0; i < x.length; i += 5) {
        for (let j = 0; j < y.length; j += 5) {
            let xi = x[i];
            let yi = y[j];
            const pathX = [];
            const pathY = [];
            for (let k = 0; k < 20; k++) {
                pathX.push(xi);
                pathY.push(yi);

                const u_val = u[Math.round(yi)][Math.round(xi)];
                const v_val = v[Math.round(yi)][Math.round(xi)];

                xi += u_val * 0.1;
                yi += v_val * 0.1;

                if (xi < 0 || xi >= x.length || yi < 0 || yi >= y.length) break;
            }
            plotData[0].x.push(...pathX, null);
            plotData[0].y.push(...pathY, null);
        }
    }

    Plotly.react('plotDiv', plotData, plotLayout);
}



