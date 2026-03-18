// model.js
import * as tf from '@tensorflow/tfjs-node';

export class TinyLLM {
  constructor(vocabSize, embeddingDim) {
    this.model = tf.sequential();
    
    // The Embedding layer: turns Token IDs into vectors of numbers
    this.model.add(tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: embeddingDim,
      inputLength: 10 // Sequence length
    }));

    // A simple LSTM layer for memory (better for your CPU than a full Transformer)
    this.model.add(tf.layers.lstm({ units: 32, returnSequences: false }));

    // The Output layer: predicts the next token ID
    this.model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));
  }

  summary() {
    this.model.summary();
  }
}