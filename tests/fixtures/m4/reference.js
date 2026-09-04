// Dependency-free transcription of the pinned IMG.LY tensor utilities plus
// the development-only onnxruntime-node parity generator. The decoded input
// is RGBA because tensorHWCtoBCHW intentionally advances by four bytes.
// Source commit: 12f56cc4f2a90d624e165a715748d22efc7a1d93.
const crypto = require('crypto');
const fs = require('fs');

function bytesOf(a) { return Buffer.from(a.buffer || a, a.byteOffset || 0, a.byteLength); }
function hash(a) { return crypto.createHash('sha256').update(bytesOf(a)).digest('hex'); }
function write(path, value) { if (path) fs.writeFileSync(path, bytesOf(value)); }
function resize(a,w,h,c,nw,nh) {
  const out=new Uint8Array(nw*nh*c), sx=w/nw, sy=h/nh;
  for(let y=0;y<nh;y++) for(let x=0;x<nw;x++) {
    const px=x*sx, py=y*sy, x1=Math.max(Math.floor(px),0), x2=Math.min(Math.ceil(px),w-1), y1=Math.max(Math.floor(py),0), y2=Math.min(Math.ceil(py),h-1), dx=px-x1,dy=py-y1;
    for(let k=0;k<c;k++) { const at=(yy,xx)=>a[(yy*w+xx)*c+k]; const v=(1-dx)*(1-dy)*at(y1,x1)+dx*(1-dy)*at(y1,x2)+(1-dx)*dy*at(y2,x1)+dx*dy*at(y2,x2); out[(y*nw+x)*c+k]=Math.round(v); }
  }
  return out;
}
function tensorHWCtoBCHW(a,h,w) {
  const stride=h*w, out=new Float32Array(3*stride);
  for(let i=0,j=0;i<a.length;i+=4,j++) { out[j]=(a[i]-128)/256; out[j+stride]=(a[i+1]-128)/256; out[j+2*stride]=(a[i+2]-128)/256; }
  return out;
}
function make(w,h,seed) { const a=[]; for(let i=0;i<w*h;i++) { a.push((i*53+seed)%256,(i*97+64+seed)%256,(255-i*29+seed+256)%256,(i*71+17)%256); } return a; }
function f32Bytes(a) { return Buffer.from(a.buffer, a.byteOffset, a.byteLength); }
function f32Stats(a) { let min=Infinity,max=-Infinity,sum=0; for(const v of a){if(v<min)min=v;if(v>max)max=v;sum+=v;} return {min,max,mean:sum/a.length}; }
function compareF32(a, b) {
  if (a.length !== b.length) throw new Error(`raw length mismatch ${a.length} vs ${b.length}`);
  let max=0,sum=0;
  for(let i=0;i<a.length;i++) { const d=Math.abs(a[i]-b[i]); if(d>max)max=d; sum+=d; }
  return {max_abs:max,mean_abs:sum/a.length};
}
function cutout(rgba, alpha) {
  const cut=new Uint8Array(rgba.length);
  for(let i=0;i<rgba.length/4;i++) { cut[4*i]=rgba[4*i];cut[4*i+1]=rgba[4*i+1];cut[4*i+2]=rgba[4*i+2];cut[4*i+3]=alpha[i]; }
  return cut;
}

const fixtures=[['landscape-3x2',3,2,11],['portrait-2x3',2,3,29],['odd-5x3',5,3,47]];
const tolerance={input_tensor_max_abs:0,raw_float_max_abs:0.000001,raw_float_mean_abs:0.0000001,restored_u8_max_abs:0};
async function main() {
  const ortPath=process.env.M4_NODE_ORT, modelPath=process.env.M4_MODEL;
  if(!ortPath || !modelPath) throw new Error('M4_NODE_ORT and M4_MODEL are required for canonical parity generation');
  const artifactDir=process.env.M4_ARTIFACT_DIR;
  if(!artifactDir || !process.env.M4_RUST_RAW_BIN) throw new Error('M4_ARTIFACT_DIR and M4_RUST_RAW_BIN are required for canonical parity generation');
  if(artifactDir) fs.mkdirSync(artifactDir,{recursive:true});
  const ort=require(ortPath), session=await ort.InferenceSession.create(modelPath,{executionProviders:['cpu']});
  const modelSha256=hash(fs.readFileSync(modelPath)), records=[];
  for(const [id,w,h,seed] of fixtures) {
    const rgba=Uint8Array.from(make(w,h,seed)), resized=resize(rgba,w,h,4,1024,1024), tensor=tensorHWCtoBCHW(resized,1024,1024);
    const tensorPath=id==='landscape-3x2'&&artifactDir?`${artifactDir}/reference-landscape.tensor.f32le`:null;
    write(tensorPath,tensor);
    const out=await session.run({input:new ort.Tensor('float32',tensor,[1,3,1024,1024])}), raw=out.output.data;
    const rawBytes=f32Bytes(raw), rawPath=id==='landscape-3x2'&&artifactDir?`${artifactDir}/reference-landscape.raw.f32le`:null;
    write(rawPath,raw);
    const u8=Uint8Array.from(raw,v=>Math.max(0,Math.min(255,Math.floor(v*255)))), restored=resize(u8,1024,1024,1,w,h), cut=cutout(rgba,restored);
    const cutPath=artifactDir?`${artifactDir}/${id}.straight-alpha-cutout.rgba`:null;
    write(cutPath,cut);
    const rec={id,width:w,height:h,decoded_rgba:Array.from(rgba),tensor:{shape:[1,3,1024,1024],sha256:hash(tensor),samples:[0,1,1023,1024,1024*1024-1].flatMap(i=>[tensor[i],tensor[1024*1024+i],tensor[2*1024*1024+i]]),artifact:tensorPath&&'reference-landscape.tensor.f32le'},raw:{sha256:hash(rawBytes),...f32Stats(raw),artifact:rawPath&&'reference-landscape.raw.f32le'},restored_alpha:{sha256:hash(restored),bytes:Array.from(restored),width:w,height:h},straight_alpha_cutout:{sha256:hash(cut),bytes:Array.from(cut),width:w,height:h,artifact:cutPath&&`${id}.straight-alpha-cutout.rgba`,alpha_zero_rgb_preserved:true}};
    records.push(rec);
  }
  let rawComparison=null;
  const rustPath=process.env.M4_RUST_RAW_BIN;
  if(rustPath) {
    const rustBytes=fs.readFileSync(rustPath), rust=new Float32Array(rustBytes.buffer,rustBytes.byteOffset,rustBytes.byteLength/4);
    const nodeRawBytes=fs.readFileSync(`${artifactDir}/reference-landscape.raw.f32le`), nodeRaw=new Float32Array(nodeRawBytes.buffer,nodeRawBytes.byteOffset,nodeRawBytes.byteLength/4);
    rawComparison={rust_artifact_sha256:hash(rustBytes),node_artifact_sha256:hash(nodeRawBytes),...compareF32(nodeRaw,rust)};
    rawComparison.status=rawComparison.max_abs<=tolerance.raw_float_max_abs&&rawComparison.mean_abs<=tolerance.raw_float_mean_abs?'pass':'fail';
  }
  const result={schema:'m4.imgly-reference.v3',source_commit:'12f56cc4f2a90d624e165a715748d22efc7a1d93',runtime:'onnxruntime-node@1.21.0',model:{file:'bundle/models/isnet',sha256:modelSha256},tolerance,records,raw_comparison:rawComparison,verdict:{input_tensor:'pass-exact-all-fixtures',raw_float:rawComparison?rawComparison.status:'not-compared',restored_u8:'pass-exact-all-fixtures',cutout:'pass-exact-all-fixtures'}};
  if(rawComparison && rawComparison.status!=='pass') process.exitCode=1;
  process.stdout.write(JSON.stringify(result,null,2)+'\n');
}
main().catch(e=>{console.error(e.stack||e);process.exit(1);});
