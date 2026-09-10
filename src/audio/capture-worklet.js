// Runs on the audio thread. Forwards raw mono frames to the main thread.
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel && channel.length) this.port.postMessage(channel.slice());
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
