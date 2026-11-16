// Run with: node --test tests/test_preprocess.mjs
// These tests use synthetic pixels only: no browser, Python, model or dataset.
import assert from 'node:assert/strict';
import test from 'node:test';
import {preprocessRGBA, softmax} from '../docs/preprocess.js';

function image(width, height, color = 255, alpha = 255) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) rgba.set([color, color, color, alpha], i * 4);
  return rgba;
}

function pixel(rgba, width, x, y, color = 0, alpha = 255) {
  rgba.set([color, color, color, alpha], (y * width + x) * 4);
}

function rectangle(rgba, width, left, top, right, bottom, color = 0) {
  for (let y = top; y < bottom; y++) {
    for (let x = left; x < right; x++) pixel(rgba, width, x, y, color);
  }
}

function centerOfMass(pixels) {
  let mass = 0, xMass = 0, yMass = 0;
  pixels.forEach((value, index) => {
    mass += value;
    xMass += value * (index % 28);
    yMass += value * Math.floor(index / 28);
  });
  return {x: xMass / mass, y: yMass / mass};
}

test('blank black, white and transparent images are rejected', () => {
  for (const [color, alpha] of [[0, 255], [255, 255], [0, 0]]) {
    assert.throws(() => preprocessRGBA(image(40, 40, color, alpha), 40, 40), /还没有看到数字/);
  }
});

test('opposite background polarities produce identical model inputs', () => {
  const darkInk = image(60, 60);
  rectangle(darkInk, 60, 42, 10, 47, 50);
  const lightInk = darkInk.map((value, index) => index % 4 === 3 ? value : 255 - value);
  const dark = preprocessRGBA(darkInk, 60, 60);
  const light = preprocessRGBA(lightInk, 60, 60);
  assert.deepEqual(dark.pixels, light.pixels);
  assert.deepEqual(dark.tensor, light.tensor);
});

test('transparent background is composited onto white without losing opaque ink', () => {
  const opaque = image(60, 60);
  const transparent = image(60, 60, 0, 0);
  for (const rgba of [opaque, transparent]) rectangle(rgba, 60, 42, 10, 47, 50);
  assert.deepEqual(
    preprocessRGBA(transparent, 60, 60),
    preprocessRGBA(opaque, 60, 60),
  );
});

test('off-center narrow strokes keep their aspect ratio and move to the center', () => {
  const rgba = image(200, 180);
  rectangle(rgba, 200, 150, 20, 160, 80);
  const {pixels, tensor} = preprocessRGBA(rgba, 200, 180);
  const center = centerOfMass(pixels);
  assert.ok(Math.abs(center.x - 13.5) <= 0.5);
  assert.ok(Math.abs(center.y - 13.5) <= 0.5);
  const xs = [], ys = [];
  pixels.forEach((value, index) => {
    if (value > 127) { xs.push(index % 28); ys.push(Math.floor(index / 28)); }
  });
  const ratio = (Math.max(...xs) - Math.min(...xs) + 1) / (Math.max(...ys) - Math.min(...ys) + 1);
  assert.ok(Math.abs(ratio - 10 / 60) < 0.06);
  assert.equal(tensor.length, 784);
  assert.ok(tensor instanceof Float32Array);
  assert.ok(tensor.every(Number.isFinite));
  assert.ok(Math.abs(tensor[0] - (0 - 0.1307) / 0.3081) < 1e-6);
});

test('negative half-pixel shifts use ties-to-even rounding', () => {
  const rgba = image(60, 60);
  // A 20x20 bounding box keeps scaling at 1. Nine pixels at its left
  // edge and ten at its right give center x=14 after initial placement.
  // The requested shift is -0.5, which must round to 0, not -2.
  for (let y = 10; y < 19; y++) pixel(rgba, 60, 20, y);
  for (let y = 20; y < 30; y++) pixel(rgba, 60, 39, y);
  assert.equal(centerOfMass(preprocessRGBA(rgba, 60, 60).pixels).x, 14);
});

test('half-pixel dimensions round to an even size', () => {
  const rgba = image(80, 80);
  rectangle(rgba, 80, 35, 20, 40, 60);
  // Fitting 5x40 to a 20-pixel box produces width 2.5 -> 2.
  const {pixels} = preprocessRGBA(rgba, 80, 80);
  const columns = new Set();
  pixels.forEach((value, index) => { if (value > 0) columns.add(index % 28); });
  assert.equal(columns.size, 2);
});

test('softmax stays finite for large scores and preserves the winning digit', () => {
  const probabilities = softmax([10000, 10001, 9999]);
  assert.ok(probabilities.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.equal(probabilities.indexOf(Math.max(...probabilities)), 1);
  assert.deepEqual(softmax([0, 0, 0, 0]), [0.25, 0.25, 0.25, 0.25]);
});
