import {preprocessRGBA, softmax} from './preprocess.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas');
const context = canvas.getContext('2d', {willReadFrequently: true});
const state = {ready: false, busy: false, hasInk: false, drawing: false, revision: 0, source: null};
let session;
const exampleButtons = Array.from(document.querySelectorAll('[data-example]'));
const probabilityRows = Array.from({length: 10}, (_, digit) => {
  const row = document.createElement('div');
  row.className = 'prob-row';
  row.innerHTML = `<span>${digit}</span><div class="prob-track"><div class="prob-fill"></div></div><span class="prob-value">—</span>`;
  $('prob-list').appendChild(row);
  return row;
});

function message(text, error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('error', error);
}

function updateControls() {
  $('predict').disabled = !state.ready || state.busy || !state.hasInk;
  $('clear').disabled = state.busy;
  $('upload').disabled = state.busy;
  exampleButtons.forEach(button => { button.disabled = state.busy; });
  $('predict-label').textContent = state.busy ? '正在识别…' : '识别数字';
  $('result-card').setAttribute('aria-busy', String(state.busy));
  canvas.style.pointerEvents = state.busy ? 'none' : 'auto';
  $('draw-shell').classList.toggle('has-ink', state.hasInk);
}

function resetResult() {
  $('guess').textContent = '—';
  $('guess-label').textContent = state.hasInk ? '准备好后，点击识别' : '等待你的第一笔';
  $('confidence').textContent = '—';
  $('processed').hidden = true;
  $('processed').removeAttribute('src');
  $('pixel-empty').hidden = false;
  probabilityRows.forEach(row => {
    row.classList.remove('winner');
    row.querySelector('.prob-fill').style.width = '0%';
    row.querySelector('.prob-value').textContent = '—';
    row.removeAttribute('aria-label');
  });
}

function clearCanvas() {
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  state.hasInk = false;
  state.drawing = false;
  state.source = null;
  state.revision++;
  resetResult();
  updateControls();
  message(state.ready ? '写下一个数字，点击「识别数字」。' : '正在载入模型，你可以先写下一个数字。');
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height};
}

canvas.addEventListener('pointerdown', event => {
  if (state.busy || (event.pointerType === 'mouse' && event.button !== 0)) return;
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  const point = canvasPoint(event);
  // Starting to draw after loading an example/image begins a fresh drawing.
  // Otherwise the upload's dark background would become part of the strokes.
  if (state.source) clearCanvas();
  state.drawing = true;
  state.hasInk = true;
  state.source = null;
  state.revision++;
  context.strokeStyle = '#181e25';
  context.fillStyle = '#181e25';
  context.lineWidth = 18;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.beginPath();
  context.arc(point.x, point.y, 9, 0, Math.PI * 2);
  context.fill();
  context.beginPath();
  context.moveTo(point.x, point.y);
  resetResult();
  updateControls();
  message(state.ready ? '写好后，点击「识别数字」。' : '可以先写数字，模型准备好后再识别。');
});
canvas.addEventListener('pointermove', event => {
  if (!state.drawing) return;
  event.preventDefault();
  const point = canvasPoint(event);
  context.lineTo(point.x, point.y);
  context.stroke();
});
['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => {
  canvas.addEventListener(name, () => { state.drawing = false; });
});
$('clear').addEventListener('click', clearCanvas);
$('upload').addEventListener('click', () => $('file').click());

async function loadImage(url, label) {
  const revision = ++state.revision;
  const image = new Image();
  image.src = url;
  await image.decode();
  if (revision !== state.revision || state.busy) return;
  if (image.width < 2 || image.height < 2) throw new Error('图片太小，请选择一张清晰的数字图片。');
  if (image.width * image.height > 16000000) throw new Error('图片尺寸太大，请先缩小到 1600 万像素以内。');
  clearCanvas();
  const scale = Math.min(260 / image.width, 260 / image.height);
  const width = image.width * scale, height = image.height * scale;
  context.drawImage(image, (280 - width) / 2, (280 - height) / 2, width, height);
  // Preserve the image's own border for black/white background detection.
  const source = document.createElement('canvas');
  const sourceScale = Math.min(1, 512 / Math.max(image.width, image.height));
  source.width = Math.max(1, Math.round(image.width * sourceScale));
  source.height = Math.max(1, Math.round(image.height * sourceScale));
  const sourceContext = source.getContext('2d', {willReadFrequently: true});
  sourceContext.fillStyle = '#ffffff';
  sourceContext.fillRect(0, 0, source.width, source.height);
  sourceContext.drawImage(image, 0, 0, source.width, source.height);
  state.source = source;
  state.hasInk = true;
  resetResult();
  updateControls();
  message(`${label}已载入，点击「识别数字」。`);
}

$('file').addEventListener('change', async event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file || state.busy) return;
  if (!/^image\/(png|jpeg|webp|bmp)$/.test(file.type)) {
    message('请选择 PNG、JPG、WebP 或 BMP 图片。', true); return;
  }
  if (file.size > 4 * 1024 * 1024) {
    message('图片太大，请选择小于 4 MB 的单个数字图片。', true); return;
  }
  const url = URL.createObjectURL(file);
  try { await loadImage(url, '图片'); }
  catch (error) { message(error.message.includes('图片') ? error.message : '无法读取这张图片，请换一张试试。', true); }
  finally { URL.revokeObjectURL(url); }
});

exampleButtons.forEach(button => button.addEventListener('click', async () => {
  if (state.busy) return;
  const digit = button.dataset.example;
  try { await loadImage(new URL(`./examples/digit_${digit}.png`, import.meta.url).href, `数字 ${digit} 的测试样本`); }
  catch { message('样本加载失败，请检查网络后再试，或直接在画板写数字。', true); }
}));

function showPreview(pixels) {
  const preview = document.createElement('canvas');
  preview.width = 28; preview.height = 28;
  const previewContext = preview.getContext('2d');
  const image = previewContext.createImageData(28, 28);
  for (let i = 0; i < pixels.length; i++) {
    image.data[i * 4] = pixels[i];
    image.data[i * 4 + 1] = pixels[i];
    image.data[i * 4 + 2] = pixels[i];
    image.data[i * 4 + 3] = 255;
  }
  previewContext.putImageData(image, 0, 0);
  $('processed').src = preview.toDataURL('image/png');
  $('processed').hidden = false;
  $('pixel-empty').hidden = true;
}

$('predict').addEventListener('click', async () => {
  if (!state.ready || state.busy || !state.hasInk) return;
  state.busy = true;
  updateControls();
  message('模型正在你的浏览器里识别…');
  try {
    const source = state.source || canvas;
    const rgba = source.getContext('2d').getImageData(0, 0, source.width, source.height).data;
    const processed = preprocessRGBA(rgba, source.width, source.height);
    const input = new window.ort.Tensor('float32', processed.tensor, [1, 1, 28, 28]);
    const outputs = await session.run({images: input});
    const probabilities = softmax(outputs.logits.data);
    if (probabilities.length !== 10 || !probabilities.every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
      throw new Error('模型返回了无效结果，请刷新页面重试。');
    }
    const confidence = Math.max(...probabilities);
    const prediction = probabilities.indexOf(confidence);
    $('guess').textContent = prediction;
    $('guess-label').textContent = '模型认为这是';
    $('confidence').textContent = `${(confidence * 100).toFixed(1)}%`;
    probabilityRows.forEach((row, digit) => {
      const percentage = probabilities[digit] * 100;
      row.classList.toggle('winner', digit === prediction);
      row.querySelector('.prob-fill').style.width = `${percentage}%`;
      row.querySelector('.prob-value').textContent = `${percentage.toFixed(1)}%`;
      row.setAttribute('aria-label', `数字 ${digit}，预测分数 ${percentage.toFixed(1)}%`);
    });
    showPreview(processed.pixels);
    message(`模型猜的是 ${prediction}。和你写的一样吗？`);
  } catch (error) {
    resetResult();
    message(error.message || '识别失败，请刷新页面重试。', true);
  } finally { state.busy = false; updateControls(); }
});

async function loadMetrics() {
  try {
    const response = await fetch(new URL('./metrics.json', import.meta.url), {signal: AbortSignal.timeout(12000)});
    if (!response.ok) throw new Error('metrics unavailable');
    const metrics = await response.json();
    if (!Number.isFinite(metrics.test_accuracy)) throw new Error('invalid metrics');
    $('training-stat').replaceChildren(document.createTextNode('MNIST 测试准确率 '));
    const score = document.createElement('strong');
    score.textContent = `${(metrics.test_accuracy * 100).toFixed(2)}%`;
    $('training-stat').append(score);
    if (Number.isInteger(metrics.epochs)) $('training-stat').append(document.createTextNode(` · 训练 ${metrics.epochs} 轮`));
    if (Number.isInteger(metrics.test_size)) $('training-stat').append(document.createTextNode(` · ${metrics.test_size.toLocaleString('zh-CN')} 张测试图`));
  } catch {
    $('training-stat').textContent = '评估数据暂未载入，可通过下方链接查看报告';
  }
}

async function initialize() {
  loadMetrics();
  try {
    if (!window.ort) throw new Error('识别组件加载失败，请检查网络后刷新页面。');
    window.ort.env.wasm.numThreads = 1;
    window.ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
    session = await window.ort.InferenceSession.create(new URL('./mnist_cnn.onnx', import.meta.url).href, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    state.ready = true;
    $('status').dataset.state = 'ready';
    $('status-text').textContent = '模型已就绪';
    $('device').textContent = '浏览器 CPU · 图片不上传';
    message(state.hasInk ? '模型已就绪，点击「识别数字」。' : '模型已就绪，写下你的第一个数字吧。');
  } catch (error) {
    $('status').dataset.state = 'error';
    $('status-text').textContent = '模型载入失败';
    $('device').textContent = '请检查网络并刷新页面';
    message('模型载入失败。请使用较新的浏览器，检查网络后刷新页面重试。', true);
    console.error('Unable to load the digit model:', error);
  } finally { updateControls(); }
}

clearCanvas();
initialize();
