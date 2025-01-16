/// Webgpu adaptation of https://www.shadertoy.com/view/Xds3zN Inigo Quilez

const V3zero: vec3f = vec3f(0.0);
const V3one: vec3f = vec3f(1.0);

const V2zero: vec2f = vec2f(0.0);
const V2one: vec2f = vec2f(1.0);

const SQRT2: f32 = 1.41421356237;
const SQRT3: f32 = 1.73205080757;


fn iBox( ori: vec3f, dir: vec3f, rad: vec3f ) -> vec2f {
	let m = 1.0 / dir;
	let n = m * ori;
	let k = abs(m) * rad;
	let t1 = -n - k;
	let t2 = -n + k;
	return vec2f(max(max(t1.x, t1.y), t1.z),
				min(min(t2.x, t2.y), t2.z));
}


fn dot2( a: vec3f ) -> f32 {
	return dot(a, a);
}

fn ndot( a: vec2f, b: vec2f ) -> f32 {
	return a.x * b.x - a.y * b.y;
}

fn sdBox( p: vec3f, b: vec3f ) -> f32 {
	let d = abs(p) - b;
	return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, V3zero));
}

fn sdBoxFrame( pos: vec3f, b: vec3f, e: f32 ) -> f32 {
    let p = abs(pos) - b;
  	let q = abs(p + e) - e;

  return min(min(
      length(max(vec3(p.x, q.y, q.z),V3zero)) + min(max(p.x, max(q.y, q.z)), 0.0),
      length(max(vec3(q.x, p.y, q.z),V3zero)) + min(max(q.x, max(p.y, q.z)), 0.0)),
      length(max(vec3(q.x, q.y, p.z),V3zero)) + min(max(q.x, max(q.y, p.z)), 0.0));
}

fn sdCappedCone( p: vec3f, h: f32, r1: f32, r2: f32 ) -> f32 {
    let q = vec2f(length(p.xz), p.y);
    
    let k1 = vec2f(r2, h);
    let k2 = vec2f(r2 - r1, 2.0 * h);
	let cond = f32(q.y < 0.0);
    let ca = vec2f(q.x - min(q.x, cond * r1 + (1.0 - cond) * r2), abs(q.y) - h);
    let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    let s = 1.0 - 2.0 * f32(cb.x < 0.0 && ca.y < 0.0);
    return s*sqrt(min(dot(ca, ca), dot(cb, cb)) );
}

fn sdCappedConeExact( p: vec3f, a: vec3f, b: vec3f, ra: f32, rb: f32) -> f32 {
    let rba  = rb - ra;
    let baba = dot(b - a, b - a);
    let papa = dot(p - a, p - a);
    let paba = dot(p - a, b - a) / baba;

    let x = sqrt( papa - paba * paba * baba );

	let cond = f32(paba < 0.5);
    let cax = max(0.0, x - (cond * ra + (1.0 - cond) * rb));
    let cay = abs(paba - 0.5) - 0.5;

    let k = rba * rba + baba;
    let f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);

    let cbx = x - ra - f * rba;
    let cby = paba - f;
    
    let s = 1.0 - 2.0 * f32(cbx < 0.0 && cay < 0.0);
    
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

fn sdCappedTorus( pos: vec3f, sc: vec2f, ra: f32, rb: f32 ) -> f32 {
	let p = vec3f(abs(pos.x), pos.yz);
	var v = dot(p, p) + ra * ra;
	if(sc.y * p.x > sc.x * p.y) {
		v -= 2.0 * ra * dot(p.xy, sc);
	}
	else {
		v -= 2.0 * ra * length(p.xy);
	}
    return sqrt(v) - rb;
}

fn sdCapsule( p: vec3f, a: vec3f, b: vec3f, r: f32 ) -> f32 {
	let pa = p - a;
	let ba = b - a;
	let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
	return length(pa - ba * h) - r;
}

fn sdCone(p: vec3f,c: vec2f, h: f32) -> f32 {
    let q = h * vec2f(c.x, - c.y) / c.y;
    let w = vec2f(length(p.xz), p.y);
    
	let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    let b = w - q * vec2f(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    let k = sign(q.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
	return sqrt(d) * sign(s);
}

fn sdCylinder( p: vec3f, h: vec2f) -> f32 {
    let d = abs(vec2f(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, V2zero));
}

fn sdCylinderOriented(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let baba = dot(ba, ba);
    let paba = dot(pa, ba);

    let x = length(pa * baba - ba * paba) - r * baba;
    let y = abs(paba - baba * 0.5) - baba * 0.5;
    let x2 = x * x;
    let y2 = y * y * baba;
	var d = 0.0;
	if(max(x,y) < 0.0) {
		d = -min(x2, y2);
	} else {
		d += f32(x > 0.0) * x2;
		d += f32(y > 0.0) * y2;
	}
    return sign(d) * sqrt(abs(d)) / baba;
}

fn sdEllipsoid( p: vec3f, r: vec3f ) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

const khp = vec3f(-0.8660254, 0.5, 0.57735);
fn sdHexPrism( pos: vec3f, h: vec2f ) -> f32 {
    let q = abs(pos);

    var p = abs(pos);
    p -= vec3f(2.0 * min(dot(khp.xy, p.xy), 0.0) * khp.xy, 0.0);
    let d = vec2f(length(p.xy - vec2f(clamp(p.x, -khp.z * h.x, khp.z * h.x), h.x)) * sign(p.y - h.x), p.z-h.y );
    return min(max(d.x, d.y), 0.0) + length(max(d, V2zero));
}

fn sdHorseshoe( pos: vec3f, c: vec2f, r: f32, le: f32, w: vec2f ) -> f32 {
    var p = pos.xy;
	p.x = abs(p.x);
    let l = length(p);
    p = mat2x2(-c.x, c.y, c.y, c.x) * p;

	let p2 = p.xy;
	if(p2.y <= 0.0 || p2.x < 0.0) {
		p.x = l * sign(-c.x);
	}
	if(p2.x < 0) {
		p.y = l;
	}

    p = vec2f(p.x - le, abs(p.y - r));
    
    let q = vec2f(length(max(p, V2zero)) + min(0.0, max(p.x, p.y)), pos.z);
    let d = abs(q) - w;
    return min(max(d.x, d.y), 0.0) + length(max(d, V2zero));
}

fn sdOctahedron( pos: vec3f, s: f32 ) -> f32 {
    let p = abs(pos);
    let m = p.x + p.y + p.z - s;
    
 	var q = vec3f();
    if( 3.0 * p.x < m ) {q = p.xyz;}
    else if( 3.0 * p.y < m ) {q = p.yzx;}
    else if( 3.0 * p.z < m ) {q = p.zxy;}
    else {return m * 0.57735027;}
    let k = clamp(0.5 * (q.z - q.y + s), 0.0, s); 
    return length(vec3f(q.x, q.y - s + k, q.z - k)); 
}

const kop = vec3f(-0.9238795325, 0.3826834323, 0.4142135623 ); 
///  vec3f(sqrt(2+sqrt(2))/2, sqrt(2-sqrt(2))/2, sqrt(2)-1) 
fn sdOctogonPrism( pos: vec3f, r: f32, h: f32 ) -> f32 {

  // reflections
  var p = abs(pos);
  p -= vec3f(2.0 * min(dot(vec2f( kop.x, kop.y), p.xy), 0.0) * vec2f( kop.x, kop.y), 0.0);
  p -= vec3f(2.0 * min(dot(vec2f(-kop.x, kop.y), p.xy), 0.0) * vec2f(-kop.x, kop.y), 0.0);
  // polygon side
  p -= vec3f(clamp(p.x, -kop.z * r, kop.z * r), r, 0.0);
  let d = vec2f(length(p.xy) * sign(p.y), p.z - h);
  return min(max(d.x, d.y), 0.0) + length(max(d, V2zero));
}

fn sdPlane( p: vec3f ) -> f32 {
	return p.y;
}

fn sdPyramid( pos: vec3f, h: f32 ) -> f32{
    let m2 = h * h + 0.25;
    var p = abs(pos.xz);
	if(p.y > p.x) {
		p = p.yx;
	}
    p -= 0.5;
	
    // project into face plane (2D)
    let q = vec3f(p.y, h * pos.y - 0.5 * p.x, h * p.x + 0.5 * pos.y);
   
    let s = max(-q.x, 0.0);
    let t = clamp((q.y - 0.5 * p.y) / (m2 + 0.25), 0.0, 1.0);
    
    let a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
	let b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    
    return sqrt(((f32(min(q.y, -q.x * m2-q.y * 0.5) <= 0.0)) * min(a, b) + q.z * q.z) / m2 ) * sign(max(q.z, -pos.y));;
}

fn sdRhombus( pos: vec3f, la: f32, lb: f32, h: f32, ra: f32 ) -> f32 {
    let p = abs(pos);
    let b = vec2f(la, lb);
    let f = clamp((ndot(b, b - 2.0 * p.xz)) / dot(b, b), -1.0, 1.0);
	let q = vec2f(length(p.xz - 0.5 * b * vec2f(1.0 - f, 1.0 + f)) * sign(p.x * b.y + p.z * b.x - b.x * b.y) - ra, p.y-h);
    return min(max(q.x, q.y), 0.0) + length(max(q, V2zero));
}

fn sdRoundCone( p: vec3f, r1: f32, r2: f32, h: f32 ) -> f32 {
    let q = vec2f(length(p.xz), p.y);
    
    let b = (r1 - r2) / h;
    let a = sqrt(1.0 - b * b);
    let k = dot(q, vec2f(-b, a));
    
    if(k < 0.0) {
		return length(q) - r1;
	} 
    if(k > a * h) {
		return length(q - vec2f(0.0, h)) - r2;
	}
        
    return dot(q, vec2f(a, b)) - r1;
}

fn sdRoundConeSI( p: vec3f, a: vec3f, b: vec3f, r1: f32, r2: f32 ) -> f32 {
    // sampling independent computations (only depend on shape)
    let  ba = b - a;
    let l2 = dot(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;
    
    // sampling dependant computations
    let pa = p - a;
    let y = dot(pa, ba);
    let z = y - l2;
    let x2 = dot2(pa * l2 - ba * y);
    let y2 = y * y * l2;
    let z2 = z * z * l2;

    // single square root!
    let k = sign(rr) * rr * rr * x2;
    if(sign(z) * a2 * z2 > k) {
		return sqrt(x2 + z2) * il2 - r2;
	}
    if(sign(y) * a2 * y2 < k) {
		return sqrt(x2 + y2) * il2 - r1;
	}
	return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

fn sdSolidAngle( pos: vec3f, c: vec2f, ra: f32 ) -> f32 {
	let p = vec2f(length(pos.xz), pos.y);
	let l = length(p) - ra;
	let m = length(p - c * clamp(dot(p, c), 0.0, ra));
	return max(l, m * sign(c.y * p.x - c.x * p.y));
}

fn sdSphere( p: vec3f, s: f32 ) -> f32 {
	return length(p) - s;
}

fn sdTorus( p: vec3f, t: vec2f ) -> f32{
    return length(vec2f(length(p.xz)-t.x, p.y)) - t.y;
}

fn sdTriPrism( pos: vec3f, hei: vec2f ) -> f32 {
	var p = pos.xy;
	var h = hei.xy;
    h.x *= 0.5 * SQRT3;
    p.x /= h.x;
    p.y /= h.x;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / SQRT3;
    if(p.x + SQRT3 * p.y > 0.0) {
		p = vec2f(p.x - SQRT3 * p.y, -SQRT3 * p.x - p.y) / 2.0;
	}
    p.x -= clamp(p.x, -2.0, 0.0);
    let d1 = length(p) * sign(-p.y) * h.x;
    let d2 = abs(pos.z) - h.y;
    return length(max(vec2f(d1, d2), V2zero)) + min(max(d1, d2), 0.0);
}

fn udTriangle( p: vec3f, a: vec3f, b: vec3f, c: vec3f ) -> f32 {
	let ba = b - a;
	let pa = p - a;
	let cb = c - b;
	let pb = p - b;
	let ac = a - c;
	let pc = p - c;
	let nor = cross(ba, ac);

	if((sign(dot(cross(ba, nor), pa)) + 
		sign(dot(cross(cb, nor), pb)) + 
		sign(dot(cross(ac, nor), pc)) < 2.0)) {

		return sqrt(
			min( min(
			dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
			dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
			dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
		); 
	}

	return sqrt(dot(nor,pa)*dot(nor,pa)/dot2(nor) );
}


fn opUnion( d1: f32, d2: f32) -> f32 {
	return min(d1, d2);
}

fn opSubstraction( d1: f32, d2: f32 ) -> f32 {
	return max(-d1, d2);
}

fn opIntersection( d1: f32, d2: f32 ) -> f32 {
	return max(d1, d2);
}

fn opXor( d1: f32, d2: f32 ) -> f32 {
	return max(min(d1, d2), -max(d1, d2));
}





