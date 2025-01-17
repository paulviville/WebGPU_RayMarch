import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { Loader } from './loader.js';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js'; 

const canvas = document.getElementById('webGpuCanvas');

const stats = new Stats()
document.body.appendChild( stats.dom );

// const gui = new GUI({});
class Raymarcher {
	#animation = false;
    #animationFrameRequest = null;

    /**
     * @param gpu WebGPU gpu
     * @param adapter WebGPU adapter
     * @param device WebGPU device
     * @param context WebGPU context of a HTMLCanvasElement
     * @param canvas {HTMLCanvasElement} Canvas element that will be used to trigger key events
     */
    constructor(gpu, adapter, device, context, canvas) {
        this.gpu = gpu;
        this.adapter = adapter;
        this.device = device;
        this.context = context;
        this.canvas = canvas;

		this.framebuffer = device.createTexture({
			label: 'framebuffer',
			size: [canvas.width, canvas.height],
			format: 'rgba16float',
			usage:
				GPUTextureUsage.RENDER_ATTACHMENT |
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING, 
		});

		this.camera = new THREE.PerspectiveCamera( 50, canvas.width / canvas.height, 0.01, 50 );
		this.camera.position.set( 2, 2, 6 );
		this.controler = new OrbitControls(this.camera, canvas);
		

		this.initGui();
    }

	initGui() {
		this.uniformParams = {
			min_dist: 0.1,
			max_dist: 20.0,
			max_steps: 50,
			max_steps_2nd: 16,
		}

		this.gui = new GUI();
		this.guiUniforms = this.gui.addFolder("uniforms");
		this.guiUniforms.add(this.uniformParams, 'min_dist').name("min dist").min(0.05).max(1.0).step(0.05);
		this.guiUniforms.add(this.uniformParams, 'max_dist').name("max dist").min(1.05).max(40.0).step(0.05);
		this.guiUniforms.add(this.uniformParams, 'max_steps').name("max steps").min(10).max(200).step(1);
		this.guiUniforms.add(this.uniformParams, 'max_steps_2nd').name("max steps 2nd").min(5).max(100).step(1);

	}

	static async boot(canvas) {
		const gpu = navigator.gpu;
		if(!gpu) {
			throw new Error("WebGPU not supported on this browser.");
		}

		const adapter = await gpu.requestAdapter();
		if(!adapter) {
			throw new Error("No appropriate GPUAdapter found.");
		}
		
		const requiredFeatures = ['bgra8unorm-storage'];
		const device = await adapter.requestDevice({requiredFeatures});
		
		const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
		const context = canvas.getContext("webgpu");
		context.configure({
			device: device,
			format: canvasFormat,
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
		});

		
		const raymarcher = new Raymarcher(gpu, adapter, device, context, canvas);
		await raymarcher.#initializeComputePipeline();
		await raymarcher.#initDisplayPipeline();
		return raymarcher;
	}

	start() {
		this.#animation = true;
		this.animate();
	}

	stop() {
		this.#animation = false;
		if(this.#animationFrameRequest !== null) {
            cancelAnimationFrame(this.#animationFrameRequest);
			this.#animationFrameRequest = null;
		}

	}

	animate() {
		const update = _ => {
  			stats.update()
            const now = performance.now();
            const deltaTime = now - lastFrame;
            lastFrame = now;

            this.render(deltaTime);

            if (this.#animation) {
                this.#animationFrameRequest = requestAnimationFrame(update);
			}
        };

        let lastFrame = performance.now();
        update();
	}

	async render(deltaTime) {
		this.camera.updateMatrixWorld();

		const mvp = this.camera.projectionMatrix.clone().multiply(this.camera.matrixWorldInverse);
		const MVP = new Float32Array(mvp.toArray());
		const INV_MVP = new Float32Array(mvp.invert().toArray());

		const uniformF32Array = new Float32Array([
			...MVP,
			...INV_MVP,
			this.uniformParams.min_dist,
			this.uniformParams.max_dist,
			0.0, 0.0,
		]);

		const uniformU32Array = new Uint32Array([
			this.uniformParams.max_steps,
			this.uniformParams.max_steps_2nd,
			0.0,
			0.0,
		])


		// const buffer = new ArrayBuffer(160);
		// const floatView = new Float32Array(buffer);
		// const uintView = new Uint32Array(buffer);
		// floatView.set(mvp, 0);
		// floatView.set(inv_mvp, 16);
		// floatView[32] = this.uniformParams.min_dist;
		// floatView[33] = this.uniformParams.max_dist;
		// uintView[36] = this.uniformParams.max_steps;
		// uintView[37] = this.uniformParams.max_steps_2nd;
		// console.log(floatView, uintView)

		// this.device.queue.writeBuffer(
		// 	this.uniformBuffer,
		// 	0,
		// 	buffer,
		// );
		this.device.queue.writeBuffer(
			this.uniformBuffer,
			0,
			uniformF32Array,
		);

		// const uint32View = new Uint32Array(this.uniformBuffer);
		// uint32View.set(uniformU32Array, 34);
		this.device.queue.writeBuffer(
			this.uniformBuffer,
			36*4,
			uniformU32Array,
		);


		const readBuffer = this.device.createBuffer({
			size: 160,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});


		// console.log(uniformF32Array, uniformU32Array, this.uniformBuffer)

		const commandEncoder = this.device.createCommandEncoder();

		commandEncoder.copyBufferToBuffer(
			this.uniformBuffer,
			0,
			readBuffer,
			0,
			160,
		);


		this.runCompute(commandEncoder);
		this.runDisplay(commandEncoder);


		this.device.queue.submit([commandEncoder.finish()]);


		await readBuffer.mapAsync(GPUMapMode.READ);
		const mappedArray = new Float32Array(readBuffer.getMappedRange());
		// console.log(mappedArray)
		readBuffer.unmap();

	}

	runCompute(commandEncoder) {
		// const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(this.pipeline);
		passEncoder.setBindGroup(0, this.bindGroup);
		passEncoder.dispatchWorkgroups(
			Math.ceil(this.framebuffer.width / this.workgroup_size_x),
			Math.ceil(this.framebuffer.height / this.workgroup_size_y)
		);
		passEncoder.end();
	}

	runDisplay(commandEncoder) {
		const canvasTexture = this.context.getCurrentTexture();
		const bindGroup = this.device.createBindGroup({
			label: 'Tonemapper.bindGroup',
			layout: this.displayBindGroupLayout,
			entries: [
				{
					// input
					binding: 0,
					resource: this.framebuffer.createView(),
				},
				{
					// output
					binding: 1,
					resource: canvasTexture.createView(),
				},
			],
		});

		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(this.displayPipeline);
		passEncoder.setBindGroup(0, bindGroup);
		passEncoder.dispatchWorkgroups(
			Math.ceil(this.framebuffer.width / this.workgroup_size_x),
			Math.ceil(this.framebuffer.height / this.workgroup_size_y)
		);
		passEncoder.end();
	}

	async #initializeComputePipeline() {
		this.uniformBuffer = this.device.createBuffer({
            size: 160,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

		const bindGroupLayout = this.device.createBindGroupLayout({
			label: "raymarcher compute bindgroup layout",
			entries: [
				{
					/// framebuffer
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: {
						access: 'write-only',
						format: this.framebuffer.format,
						viewDimension: '2d',
					},
				},
				{
					/// canvas buffer
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					buffer: {},
				},
			],
		});

		this.bindGroup = this.device.createBindGroup({
			label: "raymarcher compute bindgroup",
			layout: bindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.framebuffer.createView(),
				},
				{
					binding: 1,
					resource: {
						buffer: this.uniformBuffer,
					},
				},
			],
		});

		const pipelineLayout = this.device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		});

		const primitivesCode = await new Loader().loadText('primitives.wgsl'); 
		const raymarcherCode = await new Loader().loadText('raymarcher.wgsl'); 
		const raymarcherShaderModule = this.device.createShaderModule({
				label: "raymarcher shader module",
				code: primitivesCode + raymarcherCode,
		});

		this.workgroup_size_x = 16;
		this.workgroup_size_y = 16;

		this.pipeline = this.device.createComputePipeline({
			label: "raymarcher compute pipeline",
			layout: pipelineLayout,
			compute: {
				module: raymarcherShaderModule,
				constants: {
					WORKGROUP_SIZE_X: this.workgroup_size_x,
					WORKGROUP_SIZE_Y: this.workgroup_size_y,
				},
			},
		});
	}

	async #initDisplayPipeline() {
		const displayCode = await new Loader().loadText('display.wgsl');
		const raymarcherShaderModule = this.device.createShaderModule({
			label: "display shader module",
			code: displayCode.replace('{OUTPUT_FORMAT}', this.gpu.getPreferredCanvasFormat()),
		});


		this.displayBindGroupLayout = this.device.createBindGroupLayout({
			label: 'Display bindGroupLayout',
			entries: [
			  	{
					// input
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					texture: {
						viewDimension: '2d',
					},
				},
				{
					// output
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: {
						access: 'write-only',
						format: this.gpu.getPreferredCanvasFormat(),
						viewDimension: '2d',
					},
				},
			],
		});

		// console.log(this.gpu.getPreferredCanvasFormat())

		const pipelineLayout = this.device.createPipelineLayout({
			label: 'Display pipelineLayout',
			bindGroupLayouts: [this.displayBindGroupLayout],
		});

		this.displayPipeline = this.device.createComputePipeline({
			label: 'Display pipeline',
			layout: pipelineLayout,
			compute: {
				module: raymarcherShaderModule,
				constants: {
					WORKGROUP_SIZE_X: this.workgroup_size_x,
					WORKGROUP_SIZE_Y: this.workgroup_size_y,
				},
			},
		});
	}
}

const raymarcher = await Raymarcher.boot(canvas);

// raymarcher.render(0)
raymarcher.start()