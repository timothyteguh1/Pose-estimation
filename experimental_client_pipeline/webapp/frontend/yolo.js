const PERSON_CLASS = 0;

function letterbox(video, canvasSize) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  const scale = Math.min(canvasSize / w, canvasSize / h);
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);
  const padX = Math.floor((canvasSize - nw) / 2);
  const padY = Math.floor((canvasSize - nh) / 2);

  const canvas = document.createElement("canvas");
  canvas.width = canvasSize;
  canvas.height = canvasSize;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, canvasSize, canvasSize);
  ctx.drawImage(video, padX, padY, nw, nh);

  return { canvas, ctx, scale, padX, padY };
}

function canvasToTensor(canvas) {
  const ctx = canvas.getContext("2d");
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const size = canvas.width * canvas.height;
  const floatData = new Float32Array(3 * size);
  for (let i = 0; i < size; i++) {
    floatData[i] = data[i * 4] / 255.0;
    floatData[size + i] = data[i * 4 + 1] / 255.0;
    floatData[2 * size + i] = data[i * 4 + 2] / 255.0;
  }
  return new ort.Tensor("float32", floatData, [1, 3, canvas.height, canvas.width]);
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

function nms(boxes, scores, iouThreshold) {
  const order = scores
    .map((s, i) => i)
    .sort((i1, i2) => scores[i2] - scores[i1]);
  const keep = [];
  const suppressed = new Set();
  for (const i of order) {
    if (suppressed.has(i)) continue;
    keep.push(i);
    for (const j of order) {
      if (j === i || suppressed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > iouThreshold) suppressed.add(j);
    }
  }
  return keep;
}

function decodePersonBox(output, scale, padX, padY, origW, origH, confThreshold, iouThreshold) {
  const dims = output.dims;
  const numAttrs = dims[1];
  const numBoxes = dims[2];
  const data = output.data;

  const boxes = [];
  const scores = [];
  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];
    const personScore = data[(4 + PERSON_CLASS) * numBoxes + i];
    if (personScore < confThreshold) continue;
    boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    scores.push(personScore);
  }

  if (boxes.length === 0) return null;

  const keep = nms(boxes, scores, iouThreshold);
  let bestIdx = keep[0];
  for (const i of keep) {
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }

  let [x1, y1, x2, y2] = boxes[bestIdx];
  x1 = (x1 - padX) / scale;
  y1 = (y1 - padY) / scale;
  x2 = (x2 - padX) / scale;
  y2 = (y2 - padY) / scale;
  x1 = Math.max(0, Math.min(origW, x1));
  y1 = Math.max(0, Math.min(origH, y1));
  x2 = Math.max(0, Math.min(origW, x2));
  y2 = Math.max(0, Math.min(origH, y2));

  return { box: [x1, y1, x2, y2], score: scores[bestIdx] };
}

class YoloDetector {
  // inputSize harus sama dengan resolusi ekspor model ONNX-nya.
  async load(modelUrl, inputSize) {
    this.inputSize = inputSize;
    // onnxruntime-web 1.19.2 gagal di WebGPU (Softmax head DFL YOLOv11), 1.30.0 sudah benar.
    // wasmPaths wajib diset manual untuk bundle non-default.
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    ort.env.wasm.simd = true;
    console.log("YOLO: navigator.gpu tersedia di device/browser ini:", !!navigator.gpu);
    try {
      // webgpu dicoba sendiri (bukan array fallback) supaya jelas backend mana yang dipakai.
      this.session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["webgpu"] });
      this.backend = "webgpu (PASTI, bukan dugaan)";
      console.log("YOLO: PASTI pakai webgpu.");
    } catch (err) {
      this.lastError = err.message || String(err);
      console.warn("YOLO: webgpu gagal/tidak didukung, fallback ke wasm. Error:", this.lastError);
      this.session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
      this.backend = `wasm (fallback, webgpu gagal: ${this.lastError})`;
      console.log(`YOLO: PASTI pakai wasm (CPU), numThreads=${ort.env.wasm.numThreads}`);
    }
  }

  async detect(video, confThreshold = 0.7, iouThreshold = 0.45) {
    const { canvas, scale, padX, padY } = letterbox(video, this.inputSize);
    const tensor = canvasToTensor(canvas);
    const feeds = { images: tensor };
    const results = await this.session.run(feeds);
    const output = results[this.session.outputNames[0]];
    return decodePersonBox(
      output, scale, padX, padY, video.videoWidth, video.videoHeight,
      confThreshold, iouThreshold
    );
  }
}
