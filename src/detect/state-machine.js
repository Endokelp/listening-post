// Turns per-frame probabilities into alerts. N-of-M agreement plus hysteresis
// plus a cooldown - a confuser that looks drone-ish for one frame must not fire.
export function createDetector({
  enterThreshold = 0.7,
  exitThreshold = 0.4,
  needed = 3,
  windowSize = 5,
  cooldownMs = 3000,
  onAlert = null,
} = {}) {
  let recent = [];
  let state = 'idle';
  let lastAlertMs = -Infinity;

  function countAbove(threshold) {
    return recent.reduce((n, p) => (p >= threshold ? n + 1 : n), 0);
  }

  return {
    push(p, nowMs) {
      recent = [...recent, p].slice(-windowSize);

      if (state === 'idle') {
        if (countAbove(enterThreshold) >= needed) {
          state = 'detecting';
          if (nowMs - lastAlertMs >= cooldownMs) {
            lastAlertMs = nowMs;
            const confidence = recent.reduce((a, b) => a + b, 0) / recent.length;
            onAlert?.({ at: nowMs, confidence });
          }
        }
      } else if (countAbove(exitThreshold) < needed) {
        state = 'idle';
      }

      return state;
    },
    reset() {
      recent = [];
      state = 'idle';
      lastAlertMs = -Infinity;
    },
  };
}
