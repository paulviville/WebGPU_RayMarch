struct Uniforms {
    mvp: mat4x4<f32>, 
    inv_mvp: mat4x4<f32>,
	min_dist: f32,
	max_dist: f32,
	padding0: vec2f,
	max_steps: u32,
	max_steps_2nd: u32,
	padding1: vec2u,
}

struct Ray {
  ori : vec3f,
  dir   : vec3f,
}

@group(0) @binding(0) var framebuffer : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> uniforms : Uniforms;

override WORKGROUP_SIZE_X : u32;
override WORKGROUP_SIZE_Y : u32;

override MIN_DIST: f32 = 0.1;
override MAX_DIST: f32 = 20.0;

override MAX_STEPS: u32 = 50;
override MAX_STEPS_2ND: u32 = 16;

const epsilon: f32 = 1e-5;

fn opU(d1: vec2f, d2: vec2f) -> vec2f {
	if(d1.x < d2.x) {
		return d1;
	}
	return d2;
}

fn map(pos: vec3f) -> vec2f {
	var res = vec2f(pos.y, 0.0);

	if( sdBox(pos - vec3f(-2.0, 0.3, 0.25), vec3(0.3, 0.3, 2.5)) < res.x ) {
		res = opU(res, vec2f(sdSphere(pos - vec3f(-2.0, 0.25, 0.0), 0.25), 26.9));
	  	res = opU(res, vec2f(sdRhombus((pos - vec3(-2.0, 0.25, 1.0)).xzy, 0.15, 0.25, 0.04, 0.08 ), 17.0));
		res = opU(res, vec2f(udTriangle(pos - vec3f(-2.0, 0.0, -1.0), vec3f(-0.25, 0.0, 0.0), vec3f(0.25, 0.0, 0.0), vec3f(0.0, 0.5, 0.0)), 19.9));
	}

    if( sdBox(pos - vec3f(0.0,0.3,-1.0), vec3f(0.35,0.3,2.5) ) < res.x ) {
		res = opU( res, vec2f( sdCappedTorus((pos-vec3f(0.0, 0.30, 1.0))*vec3f(1, -1, 1), vec2f(0.866025, -0.5), 0.25, 0.05), 25.0) );
 		res = opU( res, vec2f( sdBoxFrame(pos-vec3f(0.0, 0.25, 0.0), vec3f(0.3, 0.25, 0.2), 0.025), 16.9));
		res = opU( res, vec2( sdCone(pos-vec3f( 0.0,0.45,-1.0), vec2f(0.6,0.8), 0.45), 55.0));
    	res = opU( res, vec2( sdCappedCone(  pos-vec3( 0.0,0.25,-2.0), 0.25, 0.25, 0.1 ), 13.67 ) );
    	res = opU( res, vec2f( sdSolidAngle(  pos-vec3f( 0.0,0.00,-3.0), vec2f(3,4)/5.0, 0.4 ), 49.13 ) );
    }

    if( sdBox( pos-vec3f(1.0,0.3,-1.0),vec3f(0.35,0.3,2.5)) < res.x ) {
		res = opU( res, vec2( sdTorus((pos-vec3( 1.0,0.30, 1.0)).xzy, vec2(0.25,0.05) ), 7.1 ) );
    	res = opU( res, vec2( sdBox(pos-vec3( 1.0,0.25, 0.0), vec3(0.3,0.25,0.1) ), 3.0 ) );
    	res = opU( res, vec2( sdCapsule(pos-vec3( 1.0,0.00,-1.0),vec3(-0.1,0.1,-0.1), vec3(0.2,0.4,0.2), 0.1  ), 31.9 ) );
		res = opU( res, vec2( sdCylinder(pos-vec3( 1.0,0.25,-2.0), vec2(0.15,0.25) ), 8.0 ) );
    	res = opU( res, vec2( sdHexPrism(    pos-vec3( 1.0,0.2,-3.0), vec2(0.2,0.05) ), 18.4 ) );
    }

    if( sdBox( pos-vec3(-1.0,0.35,-1.0),vec3(0.35,0.35,2.5))<res.x ) {
		res = opU( res, vec2( sdPyramid(    pos-vec3(-1.0,-0.6,-3.0), 1.0 ), 13.56 ) );
		res = opU( res, vec2( sdOctahedron( pos-vec3(-1.0,0.15,-2.0), 0.35 ), 23.56 ) );
    	res = opU( res, vec2( sdTriPrism(   pos-vec3(-1.0,0.15,-1.0), vec2(0.3,0.05) ),43.5 ) );
    	res = opU( res, vec2( sdEllipsoid(  pos-vec3(-1.0,0.25, 0.0), vec3(0.2, 0.25, 0.05) ), 43.17 ) );
    	res = opU( res, vec2( sdHorseshoe(  pos-vec3(-1.0,0.25, 1.0), vec2(cos(1.3),sin(1.3)), 0.2, 0.3, vec2(0.03,0.08) ), 11.5 ) );
    }

    if( sdBox( pos-vec3(2.0,0.3,-1.0),vec3(0.35,0.3,2.5) )<res.x ) {
		res = opU( res, vec2( sdOctogonPrism(pos-vec3( 2.0,0.2,-3.0), 0.2, 0.05), 51.8 ) );
		res = opU( res, vec2( sdCylinderOriented(    pos-vec3( 2.0,0.14,-2.0), vec3(0.1,-0.1,0.0), vec3(-0.2,0.35,0.1), 0.08), 31.2 ) );
		res = opU( res, vec2( sdCappedConeExact(  pos-vec3( 2.0,0.09,-1.0), vec3(0.1,0.0,0.0), vec3(-0.2,0.40,0.1), 0.15, 0.05), 46.1 ) );
		res = opU( res, vec2( sdRoundConeSI(   pos-vec3( 2.0,0.15, 0.0), vec3(0.1,0.0,0.0), vec3(-0.1,0.35,0.1), 0.15, 0.05), 51.7 ) );
		res = opU( res, vec2( sdRoundCone(   pos-vec3( 2.0,0.20, 1.0), 0.2, 0.1, 0.3 ), 37.0 ) );
    }
	return res;
}


fn calcNormal(pos: vec3f) -> vec3f {
	let e = vec2f(1.0, -1.0)*0.5773*0.0005;
	return normalize(
		e.xyy*map(pos+e.xyy).x +
		e.yyx*map(pos+e.yyx).x +
		e.yxy*map(pos+e.yxy).x +
		e.xxx*map(pos+e.xxx).x 
	);
}

fn calcAO(pos: vec3f, nor: vec3f) -> f32 {
	var occ = 0.0;
	var sca = 1.0;

	for(var i: u32 = 0; i < 5 && occ <= 0.35; i += 1) {
		let h = 0.01 + 0.12*f32(i)/4.0;
		let d = map(pos + h * nor).x;
		occ +=(h-d)*sca;
		sca *= 0.95;
	}
	return clamp(1.0 - 3.0 * occ, 0.0, 1.0) * (0.5 + 0.5 * nor.y);
}

fn raycast(ray: ptr<function, Ray>) -> vec2f {
	var res = vec2f(-1.0, -1.0);
	
	// var tmin = MIN_DIST;
	var tmin = uniforms.min_dist;
	var tmax = uniforms.max_dist;

	/// raytrace floor plane
	let tp1 = (0.0 - (*ray).ori.y) / (*ray).dir.y;
	if(tp1 > 0.0) {
		tmax = min(tmax, tp1);
		res = vec2f(tp1, 1.0);
	}
	
	let tb = iBox((*ray).ori - vec3f(0.0, 0.4, -0.5), (*ray).dir, vec3f(2.5, 0.41, 3.0));
	if(tb.x < tb.y && tb.y > 0.0 && tb.x < tmax) {
		tmin = max(tb.x, tmin);
		tmax = min(tb.y, tmax);

		var t = tmin;
		for(var i: u32 = 0; i < uniforms.max_steps && t < tmax; i += 1) {
			let h = map( (*ray).ori + t * (*ray).dir);
			if(abs(h.x) < 0.0001*t) {
				res = vec2(t, h.y);
				break;
			}
			t += h.x;
		}
	}

	return res; 
}

fn calcSoftShadow( ori: vec3f, dir: vec3f, mint: f32, maxt: f32 ) -> f32 {
	var tmax = maxt;
    let tp = (0.8 - ori.y) / dir.y;
	if(tp > 0.0) {
		tmax = min(tmax, tp);
	}

    var res = 1.0;
    var t = mint;
    for(var i: u32 = 0; i< uniforms.max_steps_2nd && res > 0.004 && t < tmax; i += 1)    {
		let h = map(ori + dir * t).x;
        let s = clamp(8.0 * h / t, 0.0, 1.0);
        res = min(res, s);
        t += clamp(h, 0.01, 0.2);
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

fn checkersGradBox(p: vec2f, dpdx: vec2f, dpdy: vec2f ) -> f32 {
    // filter kernel
    let w = abs(dpdx) + abs(dpdy) + 0.001;
    // analytical integral (box filter)
 	let i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    // xor pattern
    return 0.5 - 0.5*i.x*i.y;                  
}

fn render(ray: ptr<function, Ray>, rdx: vec3f, rdy: vec3f) -> vec3f {
	/// backgroud color
	var col = vec3f(0.7, 0.7, 0.9) - max((*ray).dir.y, 0.0) * 0.3;

	var res = raycast(ray);
	let t = res.x;
	let m = res.y;

	if(m > -0.5) {
		let pos = (*ray).ori + t * (*ray).dir;
		var nor = vec3f();

		col = 0.2 + 0.2*sin(m*2.0 + vec3f(0.0, 1.0, 2.0));
		var ks = 1.0;

		if(m < 1.5 ) {
			nor.y = 1.0;
			let dpdx = (*ray).ori.y*((*ray).dir/(*ray).dir.y-rdx/rdx.y);
            let dpdy = (*ray).ori.y*((*ray).dir/(*ray).dir.y-rdy/rdy.y);
			let f = checkersGradBox( 3.0*pos.xz, 3.0*dpdx.xz, 3.0*dpdy.xz );
            col = 0.15 + f*vec3f(0.05);
            ks = 0.4;
		}
		else {
			nor = calcNormal(pos);
		}
		let refl = reflect((*ray).dir, nor);

		let occ = calcAO(pos, nor);
		var lin = vec3f();

		/// Sun
		{
			let lig = normalize(vec3f(-0.5, 0.4, 0.6));
			let hal = normalize(lig - (*ray).dir);
			var dif = clamp(dot(nor, lig), 0.0, 1.0);
			dif *= calcSoftShadow(pos, lig, 0.02, 2.5);

			var spe = pow(clamp(dot(nor, hal), 0.0, 1.0), 16.0);
				spe *= dif;
				spe *= 0.04+0.94*pow(clamp(1.0-dot(hal, lig), 0.0, 1.0), 5.0);

			lin += col *2.2*dif*vec3f(1.3, 1.0, 0.7);
			lin += 5.0*spe*vec3f(1.3, 1.0, 0.7) * ks;
		}
		// sky
        {
            var dif = sqrt(clamp( 0.5+0.5*nor.y, 0.0, 1.0 ));
				dif *= occ;
            var spe = smoothstep( -0.2, 0.2, refl.y );
                  spe *= dif;
                  spe *= 0.04+0.96*pow(clamp(1.0+dot(nor,(*ray).dir),0.0,1.0), 5.0 );
                  spe *= calcSoftShadow(pos, refl, 0.02, 2.5 );
            lin += col*0.60*dif*vec3(0.40,0.60,1.15);
            lin +=     2.00*spe*vec3(0.40,0.60,1.30)*ks;
        }
        // back
        {
        	var dif = clamp( dot( nor, normalize(vec3f(0.5,0.0,0.6))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
				dif *= occ;
        	lin += col*0.55*dif*vec3f(0.25,0.25,0.25);
        }
        // sss
        {
            var dif = pow(clamp(1.0+dot(nor,(*ray).dir),0.0,1.0),2.0);
                  dif *= occ;
        	lin += col*0.25*dif*vec3(1.00,1.00,1.00);
        }

		col = lin;
		col = mix(col, vec3f(0.7, 0.7, 0.9), 1.0-exp(-0.0001*t*t*t));
	}


	return clamp(col, V3zero, V3one);
}

fn getRay(uv: vec2f) -> Ray {
	let ndcXY = (uv - 0.5) * vec2(2, -2);
	var near = uniforms.inv_mvp * vec4f(ndcXY, 0.0, 1);
    var far = uniforms.inv_mvp * vec4f(ndcXY, 1, 1);
    near /= near.w;
    far /= far.w;

	return Ray(near.xyz, normalize(far.xyz - near.xyz));
}

const AA: u32 = 4;


fn random(p: vec2f) -> f32 {
    let k = vec2<f32>(127.1, 311.7);
    return fract(sin(dot(p, k)) * 43758.5453123);
}

fn randomVec2(p: vec2f) -> vec2f {
    let k = vec2<f32>(127.1, 311.7);
    return fract(sin(dot(p, k)) * vec2f(43758.5453123, 37585.453123));
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn main(@builtin(global_invocation_id) invocation_id : vec3u) {
	let texDims = textureDimensions(framebuffer).xy;
	if(invocation_id.x > texDims.x || invocation_id.y > texDims.y) {
		return;
	}

	let resolution = vec2f(texDims);
	let invResolution = vec2f(1.0) / resolution; 
	let aspect_ratio = resolution.x / resolution.y;
	let uv = vec2f(invocation_id.xy) / resolution;
	
	var randVec = randomVec2(uv);
	let flip = vec2f(1.0, -1.0);
	
	let px = vec2f(uv.x + invResolution.x, uv.y);
	let py = vec2f(uv.x, uv.y + invResolution.y);
	let rdx = getRay(px).dir;
	let rdy = getRay(py).dir;
	

	var sumCol = vec3f();
	for(var v: u32 = 0; v < AA; v += 1) {
		var ray = getRay(uv + 0.5 * invResolution * randVec);
		randVec = randVec.yx * flip;

		var col = pow(render(&ray, rdx, rdy), vec3(0.4545));
		sumCol += col;			
	}
	sumCol /= f32(AA);

	// sumCol = vec3f(f32(uniforms.max_steps) / 50.0);


	textureStore(framebuffer, invocation_id.xy, vec4(sumCol, 1));
}