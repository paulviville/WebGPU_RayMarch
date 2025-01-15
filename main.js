import { OrbitCamera } from '../../../common/framework/util/orbit-camera.js';
import { Loader } from '../../../common/framework/util/loader.js';
import { mat4, vec4 } from '../../../lib/gl-matrix-module.js';
import Stats from './stats.module.js';

// import raymarcherCode from './raymarcher.wgsl';

const canvas = document.getElementById('webGpuCanvas');

const stats = new Stats()
document.body.appendChild( stats.dom );


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

		this.camera = new OrbitCamera(this.canvas);
		console.log(this.camera);

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
		// this.#animationFrameRequest = requestAnimationFrame(this.animate);
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
		console.log("animate")
		// if (this.#animation) {
		// 	this.#animationFrameRequest = requestAnimationFrame(this.animate);
		// }
		// console.log(this)

		const update = _ => {
  			stats.update()
//   console.log("frame")
            const now = performance.now();
            const deltaTime = now - lastFrame;
            lastFrame = now;

            this.render(deltaTime);

            if (this.#animation) {
                this.#animationFrameRequest = requestAnimationFrame(update);
				// update();
			}
        };

        let lastFrame = performance.now();
        update();
	}

	render(deltaTime) {
		this.camera.update();
		const projection = this.camera.projection;
		const view = this.camera.view;
		const mvp = mat4.multiply(mat4.create(), projection, view);
		const inv_mvp = mat4.invert(mat4.create(), mvp);

		// vec4.create()
		// console.log(inv_mvp)

		// let ndcXY = (uv - 0.5) * vec2(2, -2);
		let near = vec4.transformMat4(vec4.create(),vec4.set(vec4.create(), 0, 0, 0, 1), inv_mvp);
		let far = vec4.transformMat4(vec4.create(),vec4.set(vec4.create(), 0, 0, 1, 1), inv_mvp);
		// var near = uniforms.inv_mvp * vec4f(ndcXY, 0.0, 1);
		// var far = uniforms.inv_mvp * vec4f(ndcXY, 1, 1);
		// near /= near.w;
		// far /= far.w;
		near = vec4.scale(vec4.create(), near, 1/near[3]);
		far = vec4.scale(vec4.create(), far, 1/far[3]);
		// console.log(near, far)
		const uniformArray = new Float32Array([
			...mvp,
			...inv_mvp,
		]);
		this.device.queue.writeBuffer(
			this.uniformBuffer,
			0,
			uniformArray,
		);

		const commandEncoder = this.device.createCommandEncoder();

		this.runCompute(commandEncoder);
		this.runDisplay(commandEncoder);

		this.device.queue.submit([commandEncoder.finish()]);
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
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

		console.log(this.context)
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

		const raymarcherCode = await new Loader().loadText('raymarcher.wgsl') 
		const raymarcherShaderModule = this.device.createShaderModule({
				label: "raymarcher shader module",
				code: raymarcherCode,
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