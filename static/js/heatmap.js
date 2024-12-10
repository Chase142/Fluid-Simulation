"use strict";
var Heatmap = /** @class */ (function () {
    function Heatmap(canvas, widthRes, heightRes, vertexShaderSource, fragmentShaderSource) {
        this.program = null;
        this.positionBuffer = null;
        this.velocityBuffer = null;
        this.pressureBuffer = null;
        this.positionData = null;
        this.velocityData = null;
        this.pressureData = null;
        this.gl = (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"));
        if (this.gl == null) {
            alert("Browser does not support WebGl");
        }
        this.width = widthRes;
        this.height = heightRes;
        if (this.gl == null) {
            alert("Unable to initialize WebGL. Your browser or machine may not support it.");
            return;
        }
        this.program = this.compileShaders(vertexShaderSource, fragmentShaderSource);
        this.initPositions();
        this.initDefaultData(1);
        this.bindBuffers();
    }
    Heatmap.prototype.initPositions = function () {
        var posI = [];
        var width = this.width;
        var height = this.height;
        for (var i = 0; i < this.width - 1; i++) {
            for (var j = 0; j < this.height - 1; j++) {
                posI.push(this.indexToCoord(i + 1, width));
                posI.push(this.indexToCoord(j, height));
                posI.push(this.indexToCoord(i, width));
                posI.push(this.indexToCoord(j, height));
                posI.push(this.indexToCoord(i, width));
                posI.push(this.indexToCoord(j + 1, height));
                posI.push(this.indexToCoord(i, width));
                posI.push(this.indexToCoord(j + 1, height));
                posI.push(this.indexToCoord(i + 1, width));
                posI.push(this.indexToCoord(j + 1, height));
                posI.push(this.indexToCoord(i + 1, width));
                posI.push(this.indexToCoord(j, height));
            }
        }
        this.positionData = new Float32Array(posI);
    };
    Heatmap.prototype.initDefaultData = function (val) {
        var defaultArray = Array(this.width * this.height * 6).map(function () { return val; });
        this.velocityData = new Float32Array(defaultArray);
        this.pressureData = new Float32Array(defaultArray);
    };
    Heatmap.prototype.bindBuffers = function () {
        this.positionBuffer = this.createBuffer(this.positionData);
        this.setAttributePointer("position", this.positionBuffer, 2, this.positionData.BYTES_PER_ELEMENT);
        this.velocityBuffer = this.createBuffer(this.velocityData);
        this.setAttributePointer("velocity", this.velocityBuffer, 1, this.velocityData.BYTES_PER_ELEMENT);
        this.pressureBuffer = this.createBuffer(this.pressureData);
        this.setAttributePointer("pressure", this.pressureBuffer, 1, this.pressureData.BYTES_PER_ELEMENT);
    };
    Heatmap.prototype.updateBuffers = function () {
        var gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.pressureBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.pressureData, gl.STREAM_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.velocityBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.velocityData, gl.STREAM_DRAW);
    };
    // Compile the shaders and create the program
    Heatmap.prototype.compileShaders = function (vshaderSource, fshaderSource) {
        var gl = this.gl;
        var vs = gl.createShader(gl.VERTEX_SHADER);
        if (!vs) {
            alert("Error initializing vertex shader");
        }
        gl.shaderSource(vs, vshaderSource);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            console.error('Error compiling vertex shader:', gl.getShaderInfoLog(vs));
            return null;
        }
        var fs = gl.createShader(gl.FRAGMENT_SHADER);
        if (!fs) {
            alert("Error initializing fragment shader");
        }
        gl.shaderSource(fs, fshaderSource);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            console.error('Error compiling fragment shader:', gl.getShaderInfoLog(fs));
            return null;
        }
        var program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Error linking program:', gl.getProgramInfoLog(program));
            return null;
        }
        gl.useProgram(program);
        return program;
    };
    // Create and bind a buffer object
    Heatmap.prototype.createBuffer = function (data) {
        var gl = this.gl;
        var buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STREAM_DRAW);
        return buffer;
    };
    // Helper function to set attribute pointer
    Heatmap.prototype.setAttributePointer = function (attribute, buffer, dim, bytesPerEl, offset) {
        if (offset === void 0) { offset = 0; }
        var gl = this.gl;
        var attributeLocation = gl.getAttribLocation(this.program, attribute);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.vertexAttribPointer(attributeLocation, dim, gl.FLOAT, false, bytesPerEl * dim, offset);
        gl.enableVertexAttribArray(attributeLocation);
    };
    // Convert index to normalized coordinates
    Heatmap.prototype.indexToCoord = function (idx, size) {
        return 2 * (idx / size) - 1;
    };
    // Render the updated data
    Heatmap.prototype.drawData = function () {
        var gl = this.gl;
        // Draw the geometry
        var numVerts = (this.width - 1) * (this.height - 1) * 6;
        gl.drawArrays(gl.TRIANGLES, 0, numVerts);
    };
    Heatmap.prototype.interleave2d = function (arr) {
        var interleaved = [];
        for (var i = 0; i < arr.length - 1; i++) {
            for (var j = 0; j < arr[0].length - 1; j++) {
                interleaved.push(arr[i + 1][j]);
                interleaved.push(arr[i][j]);
                interleaved.push(arr[i][j + 1]);
                interleaved.push(arr[i][j + 1]);
                interleaved.push(arr[i + 1][j + 1]);
                interleaved.push(arr[i + 1][j]);
            }
        }
        return new Float32Array(interleaved);
    };
    Heatmap.prototype.updateData = function (velocities, pressures) {
        this.velocityData = this.interleave2d(velocities);
        this.pressureData = this.interleave2d(pressures);
        this.updateBuffers();
    };
    return Heatmap;
}());