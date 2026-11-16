// Pure pixel processing, shared by the browser demo and offline Node tests.
// The resampling follows a Lanczos-3 filter; small rounding differences from
// Pillow are possible. Handwriting accuracy must be measured independently.

const MEAN = 0.1307;
const STD = 0.3081;

function median(values) {
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

function roundEven(value) {
  const floor = Math.floor(value);
  return value - floor === 0.5 ? floor + (Math.abs(floor % 2)) : Math.round(value);
}

function borderPixels(pixels, width, height) {
  const border = [];
  for (let x = 0; x < width; x++) border.push(pixels[x], pixels[(height - 1) * width + x]);
  for (let y = 0; y < height; y++) border.push(pixels[y * width], pixels[y * width + width - 1]);
  return border;
}

function lanczos(x) {
  x = Math.abs(x);
  if (x < 1e-9) return 1;
  if (x >= 3) return 0;
  const angle = Math.PI * x;
  return (Math.sin(angle) * Math.sin(angle / 3)) / (angle * angle / 3);
}

function coefficients(inputSize, outputSize) {
  const scale = inputSize / outputSize;
  const filterScale = Math.max(1, scale);
  return Array.from({length: outputSize}, (_, i) => {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - 3 * filterScale));
    const end = Math.min(inputSize, Math.ceil(center + 3 * filterScale));
    const weights = [];
    let total = 0;
    for (let j = start; j < end; j++) {
      const weight = lanczos((j + 0.5 - center) / filterScale);
      weights.push(weight);
      total += weight;
    }
    return {start, weights: weights.map(weight => weight / total)};
  });
}

function resize(pixels, width, height, targetWidth, targetHeight) {
  const horizontal = coefficients(width, targetWidth);
  const vertical = coefficients(height, targetHeight);
  const intermediate = new Uint8Array(targetWidth * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const {start, weights} = horizontal[x];
      let value = 0;
      for (let k = 0; k < weights.length; k++) value += pixels[y * width + start + k] * weights[k];
      intermediate[y * targetWidth + x] = Math.min(255, Math.max(0, Math.round(value)));
    }
  }
  const result = new Uint8Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y++) {
    const {start, weights} = vertical[y];
    for (let x = 0; x < targetWidth; x++) {
      let value = 0;
      for (let k = 0; k < weights.length; k++) value += intermediate[(start + k) * targetWidth + x] * weights[k];
      result[y * targetWidth + x] = Math.min(255, Math.max(0, Math.round(value)));
    }
  }
  return result;
}

export function preprocessRGBA(rgba, width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2) {
    throw new Error('图片太小，请提供一张清晰的数字图片。');
  }
  if (width * height > 16000000 || rgba.length !== width * height * 4) {
    throw new Error('图片尺寸无效或过大。');
  }
  const values = new Float64Array(width * height);
  for (let i = 0; i < values.length; i++) {
    const offset = i * 4;
    const alpha = rgba[offset + 3] / 255;
    values[i] = Math.round(alpha * (0.299 * rgba[offset] + 0.587 * rgba[offset + 1] + 0.114 * rgba[offset + 2]) + 255 * (1 - alpha));
  }
  if (median(borderPixels(values, width, height)) > 127) {
    for (let i = 0; i < values.length; i++) values[i] = 255 - values[i];
  }
  const background = median(borderPixels(values, width, height));
  let peak = 0;
  for (let i = 0; i < values.length; i++) {
    values[i] = Math.max(0, values[i] - background);
    peak = Math.max(peak, values[i]);
  }
  if (peak < 25) throw new Error('还没有看到数字，请先写一个 0～9 的数字。');
  let xMin = width, xMax = -1, yMin = height, yMax = -1, count = 0;
  const threshold = Math.max(25, peak * 0.2);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (values[y * width + x] > threshold) {
        xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
        yMin = Math.min(yMin, y); yMax = Math.max(yMax, y); count++;
      }
    }
  }
  if (count < 5) throw new Error('笔迹太少，请写一个完整的数字。');
  const cropWidth = xMax - xMin + 1, cropHeight = yMax - yMin + 1;
  const crop = new Uint8Array(cropWidth * cropHeight);
  for (let y = 0; y < cropHeight; y++) {
    for (let x = 0; x < cropWidth; x++) {
      const value = values[(yMin + y) * width + xMin + x];
      crop[y * cropWidth + x] = value < peak * 0.08 ? 0 : Math.floor(Math.min(255, value * 255 / peak));
    }
  }
  const scale = 20 / Math.max(cropWidth, cropHeight);
  const targetWidth = Math.max(1, roundEven(cropWidth * scale));
  const targetHeight = Math.max(1, roundEven(cropHeight * scale));
  const ink = resize(crop, cropWidth, cropHeight, targetWidth, targetHeight);
  const pixels = new Uint8Array(28 * 28);
  const left = Math.floor((28 - targetWidth) / 2), top = Math.floor((28 - targetHeight) / 2);
  let mass = 0, xMass = 0, yMass = 0;
  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const value = ink[y * targetWidth + x];
      pixels[(top + y) * 28 + left + x] = value;
      mass += value; xMass += value * (left + x); yMass += value * (top + y);
    }
  }
  if (!mass) throw new Error('笔迹太细，请写得粗一些后再试。');
  const dx = roundEven(13.5 - xMass / mass), dy = roundEven(13.5 - yMass / mass);
  const centered = new Uint8Array(28 * 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      if (x + dx >= 0 && x + dx < 28 && y + dy >= 0 && y + dy < 28) {
        centered[(y + dy) * 28 + x + dx] = pixels[y * 28 + x];
      }
    }
  }
  const tensor = Float32Array.from(centered, value => (value / 255 - MEAN) / STD);
  return {pixels: centered, tensor};
}

export function softmax(logits) {
  const maximum = Math.max(...logits);
  const exponential = Array.from(logits, value => Math.exp(value - maximum));
  const total = exponential.reduce((sum, value) => sum + value, 0);
  return exponential.map(value => value / total);
}
