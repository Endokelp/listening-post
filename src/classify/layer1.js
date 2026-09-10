import weights from './layer1-weights.json' with { type: 'json' };

export const DEFAULT_WEIGHTS = weights;

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export function classify(features, w = DEFAULT_WEIGHTS) {
  let z = w.intercept;
  for (const [name, coefficient] of Object.entries(w.coefficients)) {
    z += coefficient * (features[name] ?? 0);
  }
  return sigmoid(z);
}
