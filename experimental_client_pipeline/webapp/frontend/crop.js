const TRAINING_ASPECT_RATIO = 1080 / 1920;

function expandBbox(bbox, paddingRatio, frameW, frameH, targetAspectRatio = TRAINING_ASPECT_RATIO) {
  let [x1, y1, x2, y2] = bbox;
  if (paddingRatio > 0) {
    const padX = (x2 - x1) * paddingRatio;
    const padY = (y2 - y1) * paddingRatio;
    x1 -= padX; y1 -= padY; x2 += padX; y2 += padY;
  }

  const boxW = x2 - x1;
  const boxH = y2 - y1;
  if (boxW > 0 && boxH > 0) {
    const currentRatio = boxW / boxH;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    if (currentRatio < targetAspectRatio) {
      const newW = boxH * targetAspectRatio;
      x1 = cx - newW / 2; x2 = cx + newW / 2;
    } else if (currentRatio > targetAspectRatio) {
      const newH = boxW / targetAspectRatio;
      y1 = cy - newH / 2; y2 = cy + newH / 2;
    }
  }

  x1 = Math.max(0, Math.round(x1));
  y1 = Math.max(0, Math.round(y1));
  x2 = Math.min(frameW, Math.round(x2));
  y2 = Math.min(frameH, Math.round(y2));
  return [x1, y1, x2, y2];
}

const BODY_CONNECTIONS = [
  [0, 11], [0, 12], [11, 12], [23, 24],
  [11, 23], [12, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [29, 31], [27, 31],
  [28, 30], [30, 32], [28, 32],
];
