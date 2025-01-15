/// Webgpu adaptation of https://www.shadertoy.com/view/Xds3zN Inigo Quilez

struct Uniforms {
    mvp: mat4x4<f32>, 
    inv_mvp: mat4x4<f32>, 
}

struct Ray {
  ori : vec3f,
  dir   : vec3f,
}


@group(0) @binding(0) var framebuffer : texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> uniforms : Uniforms;

override WORKGROUP_SIZE_X : u32;
override WORKGROUP_SIZE_Y : u32;

const SQRT2: f32 = 1.41421356237;
const SQRT3: f32 = 1.73205080757;

fn dot2(a: vec3f) -> f32 {
	return dot(a, a);
}

fn ndot(a: vec2f, b: vec2f) -> f32 {
	return a.x*b.x - a.y*b.y;
}

fn sdSphere(p: vec3f, s: f32) -> f32 {
	return length(p) - s;
}

fn sdPlane(p: vec3f) -> f32 {
	return p.y;
}

fn sdBox(p: vec3f, b: vec3f) -> f32 {
	let d = abs(p) - b;
	return min(max(d.x, max(d.y,d.z)), 0.0) + length(max(d, V3zero));
}

fn sdSolidAngle(pos: vec3f, c: vec2f, ra: f32) -> f32 {
	let p = vec2f(length(pos.xz), pos.y);
	let l = length(p) - ra;
	let m = length(p - c*clamp(dot(p,c), 0.0, ra));
	return max(l, m*sign(c.y*p.x-c.x*p.y));
}

fn sdRhombus(pos: vec3f, la: f32, lb: f32, h: f32, ra: f32) -> f32 {
    let p = abs(pos);
    let b = vec2f(la,lb);
    let f = clamp( (ndot(b,b-2.0*p.xz))/dot(b,b), -1.0, 1.0 );
	let q = vec2f(length(p.xz-0.5*b*vec2f(1.0-f,1.0+f))*sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h);
    return min(max(q.x,q.y),0.0) + length(max(q,V2zero));
}

fn sdCappedTorus(pos: vec3f, sc: vec2f, ra: f32, rb: f32) -> f32 {
	let p = vec3f(abs(pos.x), pos.yz);
	var v = dot(p,p) + ra*ra;
	if(sc.y*p.x>sc.x*p.y) {
		v -= 2.0 * ra * dot(p.xy,sc);
	}
	else {
		v -= 2.0 * ra * length(p.xy);
	}
    return sqrt(v) - rb;
}

fn sdTorus(p: vec3f, t: vec2f) -> f32{
    return length( vec2(length(p.xz)-t.x,p.y) )-t.y;
}

fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32 ) -> f32 {
	let pa = p-a;
	let ba = b-a;
	let h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h ) - r;
}

fn sdBoxFrame(pos: vec3f, b: vec3f, e: f32 ) -> f32 {
    let p = abs(pos) - b;
  	let q = abs(p + e) - e;

  return min(min(
      length(max(vec3(p.x,q.y,q.z),V3zero))+min(max(p.x,max(q.y,q.z)),0.0),
      length(max(vec3(q.x,p.y,q.z),V3zero))+min(max(q.x,max(p.y,q.z)),0.0)),
      length(max(vec3(q.x,q.y,p.z),V3zero))+min(max(q.x,max(q.y,p.z)),0.0));
}

fn sdCone(p: vec3f,c: vec2f, h: f32) -> f32 {
    let q = h * vec2f(c.x, - c.y) / c.y;
    let w = vec2f(length(p.xz), p.y);
    
	let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    let b = w - q * vec2f(clamp(w.x / q.x, 0.0, 1.0 ), 1.0);
    let k = sign(q.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
	return sqrt(d) * sign(s);
}

fn sdCappedCone(p: vec3f, h: f32, r1: f32, r2: f32) -> f32 {
    let q = vec2f( length(p.xz), p.y );
    
    let k1 = vec2f(r2,h);
    let k2 = vec2f(r2-r1,2.0*h);
	let cond = f32(q.y < 0.0);
    let ca = vec2f(q.x-min(q.x,cond * r1 + (1.0 - cond) * r2), abs(q.y)-h);
    let cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot(k2, k2), 0.0, 1.0 );
    let s = 1.0 - 2.0 * f32(cb.x < 0.0 && ca.y < 0.0);
    return s*sqrt( min(dot(ca, ca), dot(cb, cb)) );
}

fn sdCappedConeExact(p: vec3f, a: vec3f, b: vec3f, ra: f32, rb: f32) -> f32 {
    let rba  = rb-ra;
    let baba = dot(b-a,b-a);
    let papa = dot(p-a,p-a);
    let paba = dot(p-a,b-a)/baba;

    let x = sqrt( papa - paba*paba*baba );

	let cond = f32(paba<0.5);
    let cax = max(0.0,x-(cond * ra + (1.0 - cond)* rb));
    let cay = abs(paba-0.5)-0.5;

    let k = rba*rba + baba;
    let f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );

    let cbx = x-ra - f*rba;
    let cby = paba - f;
    
    let s = 1.0 - 2.0 * f32(cbx < 0.0 && cay < 0.0);
    
    return s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
}

fn sdCylinder( p: vec3f, h: vec2f) -> f32 {
    let d = abs(vec2f(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,V2zero));
}

fn sdCylinderOriented(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let baba = dot(ba,ba);
    let paba = dot(pa,ba);

    let x = length(pa*baba-ba*paba) - r*baba;
    let y = abs(paba-baba*0.5)-baba*0.5;
    let x2 = x*x;
    let y2 = y*y*baba;
	var d = 0.0;
	if(max(x,y)<0.0) {
		d = -min(x2, y2);
	} else {
		d += f32(x > 0.0) * x2;
		d += f32(y > 0.0) * y2;
	}
    return sign(d)*sqrt(abs(d))/baba;
}

fn sdPyramid(pos: vec3f, h: f32 ) -> f32{
    let m2 = h*h + 0.25;
    var p = abs(pos.xz);
	if(p.y > p.x) {
		p = p.yx;
	}
    p -= 0.5;
	
    // project into face plane (2D)
    let q = vec3f( p.y, h*pos.y - 0.5*p.x, h*p.x + 0.5*pos.y);
   
    let s = max(-q.x,0.0);
    let t = clamp( (q.y-0.5*p.y)/(m2+0.25), 0.0, 1.0 );
    
    let a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
	let b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
    
	let cond1 = f32(min(q.y,-q.x*m2-q.y*0.5) > 0.0);
    let d2 =  (1.0 - cond1) * min(a,b);
    
    // recover 3D and scale, and add sign
    return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-pos.y));;
}

fn sdRoundCone( p: vec3f, r1: f32, r2: f32, h: f32 ) -> f32 {
    let q = vec2( length(p.xz), p.y );
    
    let b = (r1-r2)/h;
    let a = sqrt(1.0-b*b);
    let k = dot(q,vec2(-b,a));
    
    if( k < 0.0 ) {
		return length(q) - r1;
	} 
    if( k > a*h ) {
		return length(q-vec2(0.0,h)) - r2;
	}
        
    return dot(q, vec2(a,b) ) - r1;
}

fn sdRoundConeSI(p: vec3f, a: vec3f, b: vec3f, r1: f32, r2: f32) -> f32 {
    // sampling independent computations (only depend on shape)
    let  ba = b - a;
    let l2 = dot(ba,ba);
    let rr = r1 - r2;
    let a2 = l2 - rr*rr;
    let il2 = 1.0/l2;
    
    // sampling dependant computations
    let pa = p - a;
    let y = dot(pa,ba);
    let z = y - l2;
    let x2 = dot2( pa*l2 - ba*y );
    let y2 = y*y*l2;
    let z2 = z*z*l2;

    // single square root!
    let k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2 > k ) {
		return  sqrt(x2 + z2) * il2 - r2;
	}
    if( sign(y)*a2*y2 < k ) {
		return  sqrt(x2 + y2) * il2 - r1;
	}
	return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}


fn sdTriPrism(pos: vec3f, hei:vec2f ) -> f32 {
	var p = pos.xy;
	var h = hei.xy;
    h.x *= 0.5*SQRT3;
    p.x /= h.x;
    p.y /= h.x;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/SQRT3;
    if( p.x+SQRT3*p.y>0.0 ) {
		p=vec2(p.x-SQRT3*p.y,-SQRT3*p.x-p.y)/2.0;
	}
    p.x -= clamp( p.x, -2.0, 0.0 );
    let d1 = length(p)*sign(-p.y)*h.x;
    let d2 = abs(pos.z)-h.y;
    return length(max(vec2f(d1,d2), V2zero)) + min(max(d1,d2), 0.);
}

fn sdOctahedron(pos: vec3f, s: f32) -> f32 {
    let p = abs(pos);
    let m = p.x + p.y + p.z - s;
    
 	var q = vec3f();
    if( 3.0*p.x < m ) {q = p.xyz;}
    else if( 3.0*p.y < m ) {q = p.yzx;}
    else if( 3.0*p.z < m ) {q = p.zxy;}
    else {return m*0.57735027;}
    let k = clamp(0.5*(q.z-q.y+s),0.0,s); 
    return length(vec3f(q.x,q.y-s+k,q.z-k)); 
}

fn sdEllipsoid( p: vec3f, r: vec3f ) -> f32 {
    let k0 = length(p/r);
    let k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

fn sdHorseshoe( pos: vec3f, c: vec2f, r: f32, le: f32, w: vec2f ) -> f32 {
    var p = pos.xy;
	p.x = abs(p.x);
    let l = length(p);
    p = mat2x2(-c.x, c.y, c.y, c.x) * p;

	var p2 = p.xy;
	if(p2.y<=0.0 || p2.x<0.0) {
		p.x = l*sign(-c.x);
	}
	if(p2.x < 0) {
		p.y = l;
	}

    p = vec2f(p.x - le,abs(p.y-r));
    
    let q = vec2f(length(max(p,V2zero)) + min(0.0,max(p.x,p.y)),pos.z);
    let d = abs(q) - w;
    return min(max(d.x,d.y),0.0) + length(max(d,V2zero));
}

fn sdHexPrism( pos: vec3f, h: vec2f ) -> f32 {
    let q = abs(pos);

    let k = vec3f(-0.8660254, 0.5, 0.57735);
    var p = abs(pos);
    p -= vec3f(2.0*min(dot(k.xy, p.xy), 0.0)*k.xy, 0.0);
    let d = vec2(
       length(p.xy - vec2(clamp(p.x, -k.z*h.x, k.z*h.x), h.x))*sign(p.y - h.x),
       p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,V2zero));
}

fn sdOctogonPrism( pos: vec3f, r: f32, h: f32 ) -> f32 {
  let k = vec3f(-0.9238795325,   // sqrt(2+sqrt(2))/2 
                       0.3826834323,   // sqrt(2-sqrt(2))/2
                       0.4142135623 ); // sqrt(2)-1 
  // reflections
  var p = abs(pos);
  p -= vec3f(2.0*min(dot(vec2( k.x,k.y),p.xy),0.0)*vec2( k.x,k.y), 0.0);
  p -= vec3f(2.0*min(dot(vec2(-k.x,k.y),p.xy),0.0)*vec2(-k.x,k.y), 0.0);
  // polygon side
  p -= vec3f(clamp(p.x, -k.z*r, k.z*r), r, 0.0);
  let d = vec2f( length(p.xy)*sign(p.y), p.z-h );
  return min(max(d.x,d.y),0.0) + length(max(d,V2zero));
}

fn opU(d1: vec2f, d2: vec2f) -> vec2f {
	if(d1.x < d2.x) {
		return d1;
	}
	else {
		return d2;
	}
}


fn iBox(ori: vec3f, dir: vec3f, rad: vec3f) -> vec2f {
	let m = 1.0 / dir;
	let n = m * ori;
	let k = abs(m) * rad;
	let t1 = -n - k;
	let t2 = -n + k;
	return vec2f(max(max(t1.x, t1.y), t1.z),
				min(min(t2.x, t2.y), t2.z));
}


const MIN_DIST: f32 = 1.0;
const MAX_DIST: f32 = 20.0;

const MAX_STEPS: u32 = 70;
const MAX_STEPS_2ND: u32 = 16;
const epsilon: f32 = 1e-5;

const V3zero: vec3f = vec3f(0.0);
const V3one: vec3f = vec3f(1.0);

const V2zero: vec2f = vec2f(0.0);
const V2one: vec2f = vec2f(1.0);

fn map(pos: vec3f) -> vec2f {
	var res = vec2f(pos.y, 0.0);
	if( sdBox(pos - vec3f(-2.0, 0.3, 0.25), vec3(0.3, 0.3, 1.0)) < res.x ) {
		res = opU(res, vec2f(sdSphere(pos - vec3f(-2.0, 0.25, 0.0), 0.25), 26.9));
	  	res = opU(res, vec2f(sdRhombus((pos - vec3(-2.0, 0.25, 1.0)).xzy, 0.15, 0.25, 0.04, 0.08 ), 17.0));
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
	
	var tmin = MIN_DIST;
	var tmax = MAX_DIST;

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
		for(var i: u32 = 0; i < MAX_STEPS && t < tmax; i += 1) {
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

fn calcSoftShadow(ori: vec3f, dir: vec3f, mint: f32, maxt: f32 ) -> f32 {
	var tmax = maxt;
    let tp = (0.8-ori.y)/dir.y;
	if( tp>0.0 ) {
		tmax = min( tmax, tp );
	}

    var res = 1.0;
    var t = mint;
    for(var i: u32 = 0; i< MAX_STEPS_2ND && res > 0.004 && t < tmax; i += 1)    {
		let h = map( ori + dir*t ).x;
        let s = clamp(8.0*h/t,0.0,1.0);
        res = min( res, s );
        t += clamp( h, 0.01, 0.2 );
    }
    res = clamp( res, 0.0, 1.0 );
    return res*res*(3.0-2.0*res);
}

fn checkersGradBox(p: vec2f, dpdx: vec2f, dpdy: vec2f ) -> f32 {
    // filter kernel
    let w = abs(dpdx)+abs(dpdy) + 0.001;
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

const AA: u32 = 2;


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
	
	// let rand0 = 0.5 - random(vec2f(invocation_id.xy));
	// let rand1 = 0.5 - random(2.0 * vec2f(invocation_id.xy));
	var randVec = randomVec2(uv);
	let flip = vec2f(1.0, -1.0);
	
	let px = vec2f(uv.x + invResolution.x, uv.y);
	let py = vec2f(uv.x, uv.y + invResolution.y);
	let rdx = getRay(px).dir;
	let rdy = getRay(py).dir;
	

	var sumCol = vec3f();
	for(var v: u32 = 0; v < AA; v += 1) {

		// let rand1 = random(2.0 * uv + vec2f(f32(AA)));

		var ray = getRay(uv + 0.5 * invResolution * randVec);
		randVec = randVec.yx * flip;


		var col = pow(render(&ray, rdx, rdy), vec3(0.4545));
		sumCol += col;			
	}
	sumCol /= f32(AA);



	textureStore(framebuffer, invocation_id.xy, vec4(sumCol, 1));
}