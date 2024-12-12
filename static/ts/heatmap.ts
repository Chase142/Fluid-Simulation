class Heatmap {

  private program : WebGLProgram | null = null;
  private gl : WebGLRenderingContext | null;

  private positionBuffer : WebGLBuffer | null = null;
  private velocityBuffer : WebGLBuffer | null = null;
  private pressureBuffer : WebGLBuffer | null = null;

  private positionData : Float32Array | null = null;
  private velocityData : Float32Array | null = null;
  private pressureData : Float32Array | null = null;

  public width : number;
  public height : number;

  constructor(canvas : HTMLCanvasElement, widthRes : number, heightRes : number,
     vertexShaderSource : string, fragmentShaderSource : string) {
    this.gl = (canvas.getContext("webgl") || canvas.getContext("experimental-webgl")) as WebGLRenderingContext | null;

    if(this.gl == null){
      alert("Browser does not support WebGl")
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

  private initPositions(){

    const posI : number[] = []
    const width = this.width - 1;
    const height = this.height - 1;

    for (let i = 0; i < this.width; i++) {
        for (let j = 0; j < this.height; j++){
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

    this.positionData = new Float32Array(new Float32Array(posI));
  }

  private initDefaultData(val : number){
    var defaultArray = Array(this.width * this.height * 6).map(() => val);
    this.velocityData = new Float32Array(defaultArray);
    this.pressureData = new Float32Array(defaultArray);
  }

  private bindBuffers(){
    this.positionBuffer = this.createBuffer(this.positionData!)
    this.setAttributePointer("position", this.positionBuffer, 2, this.positionData!.BYTES_PER_ELEMENT);  
    this.velocityBuffer = this.createBuffer(this.velocityData!);
    this.setAttributePointer("velocity", this.velocityBuffer, 1, this.velocityData!.BYTES_PER_ELEMENT);  
    this.pressureBuffer = this.createBuffer(this.pressureData!);
    this.setAttributePointer("pressure", this.pressureBuffer, 1, this.pressureData!.BYTES_PER_ELEMENT);
  }

  private updateBuffers(){
    const gl : WebGLRenderingContext = this.gl!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pressureBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.pressureData, gl.STREAM_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.velocityBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.velocityData, gl.STREAM_DRAW);
  }

  // Compile the shaders and create the program
  private compileShaders(vshaderSource : string, fshaderSource: string) {
    const gl = this.gl!;
    const vs : WebGLShader = gl.createShader(gl.VERTEX_SHADER)!;
    if(!vs){
      alert("Error initializing vertex shader");
    }
    gl.shaderSource(vs, vshaderSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      console.error('Error compiling vertex shader:', gl.getShaderInfoLog(vs));
      return null;
    }

    const fs : WebGLShader = gl.createShader(gl.FRAGMENT_SHADER)!;
    if(!fs){
      alert("Error initializing fragment shader");
    }
    gl.shaderSource(fs, fshaderSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      console.error('Error compiling fragment shader:', gl.getShaderInfoLog(fs));
      return null;
    }

    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Error linking program:', gl.getProgramInfoLog(program));
      return null;
    }

    gl.useProgram(program);
    return program;
  }

  // Create and bind a buffer object
  private createBuffer(data : Float32Array) : WebGLBuffer {
    const gl : WebGLRenderingContext = this.gl!;
    const buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STREAM_DRAW);
    return buffer;
  }

  // Helper function to set attribute pointer
  private setAttributePointer(attribute : string, buffer : WebGLBuffer, dim : number, bytesPerEl : number, offset : number = 0) {
    const gl = this.gl!;
    const attributeLocation = gl.getAttribLocation(this.program!, attribute);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.vertexAttribPointer(attributeLocation, dim, gl.FLOAT, false, bytesPerEl * dim, offset);
    gl.enableVertexAttribArray(attributeLocation);
  }

  // Convert index to normalized coordinates
  private indexToCoord(idx : number, size : number) {
      return 2 * (idx / size) - 1;
  }

  // Render the updated data
  public drawData() {
    const gl = this.gl!;
    // Draw the geometry
    const numVerts = (this.width) * (this.height) * 6;
    gl.drawArrays(gl.TRIANGLES, 0, numVerts);
  }

  private interleave2d(arr : number[][]){
    var interleaved : number[] = []

    for (let i = 0; i < arr.length - 1; i++) {
      for (let j = 0; j < arr[0].length; j++){
          interleaved.push(arr[i + 1][j]);
          interleaved.push(arr[i][j]);
          interleaved.push(arr[i][j + 1]);
          interleaved.push(arr[i][j + 1]);
          interleaved.push(arr[i + 1][j + 1]);
          interleaved.push(arr[i + 1][j]);
      }
    }

    return new Float32Array(interleaved);
  }

  public updateData(velocities : number[][], pressures : number[][]){
    this.velocityData = this.interleave2d(velocities);
    this.pressureData = this.interleave2d(pressures);

    this.updateBuffers();
  }
}