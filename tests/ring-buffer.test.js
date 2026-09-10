import test from 'node:test';
import assert from 'node:assert/strict';
import { createRingBuffer, createWindower } from '../src/audio/ring-buffer.js';

test('ring buffer reads back what was written, in order', () => {
  const rb = createRingBuffer(16);
  rb.write(Float32Array.from([1, 2, 3, 4]));
  assert.equal(rb.available, 4);
  assert.deepEqual(Array.from(rb.read(4)), [1, 2, 3, 4]);
  assert.equal(rb.available, 0);
});

test('ring buffer drops the oldest samples when it overflows', () => {
  const rb = createRingBuffer(4);
  rb.write(Float32Array.from([1, 2, 3, 4, 5, 6]));
  assert.equal(rb.available, 4);
  assert.deepEqual(Array.from(rb.read(4)), [3, 4, 5, 6]);
});

test('windower emits overlapping windows at the hop interval', () => {
  const windows = [];
  const w = createWindower(4, 2, (win) => windows.push(Array.from(win)));
  w.push(Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]));
  assert.deepEqual(windows[0], [1, 2, 3, 4]);
  assert.deepEqual(windows[1], [3, 4, 5, 6]);
  assert.deepEqual(windows[2], [5, 6, 7, 8]);
});
