import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-cpu';

import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import path from 'path';

// Fix for __dirname in ESM
const __dirname = path.dirname(fileURLToPath(import.meta.url));

class EvieOptimizer extends tf.Optimizer {
    constructor(learningRate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        super();
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
        this.accumulators = new Map();

        this.workers = [];
        this.numCores = 4;

        this.registered_variables = tf.engine().registeredVariables;

        this.weightMap = new Map(); // 🔒 Private internal map

        // Initialize the pool
        for (let i = 0; i < this.numCores; i++) {
            // Point to your worker file
            const workerPath = path.join(__dirname, 'optimiser_worker.js');
            this.workers.push(new Worker(workerPath));
        }
    }

   getOrCreateSABs(name, rawValues, shape, size) {
    // If we already have buffers for this variable, just update the gradient
    if (this.accumulators.has(name)) {
        const sabs = this.accumulators.get(name);
        
        // Use the raw values we extracted in applyGradients
        new Float32Array(sabs.sharedGrad).set(rawValues);
        return sabs;
    }

    // First-time setup for this specific variable
    const byteLength = size * 4;

    const sabs = {
        sharedGrad: new SharedArrayBuffer(byteLength),
        sharedM: new SharedArrayBuffer(byteLength),
        sharedV: new SharedArrayBuffer(byteLength),
        sharedVar: new SharedArrayBuffer(byteLength)
    };

    // Initialize sharedVar with current weights from the registered variable
    const varView = new Float32Array(sabs.sharedVar);
    varView.set(this.registered_variables[name].dataSync());

    // Initialize m and v to zeros (standard for Adam)
    new Float32Array(sabs.sharedM).fill(0);
    new Float32Array(sabs.sharedV).fill(0);

    // Load the current gradient values
    new Float32Array(sabs.sharedGrad).set(rawValues);

    // Store in our Map so we reuse these buffers in the next iteration
    this.accumulators.set(name, sabs);

    return sabs;
}

    // Define the actual math step
    async applyGradients(variableGradients) {
    // 1. IMMEDIATELY extract data from all tensors before any 'await'
    // This prevents the "Tensor is disposed" error.
    const gradData = variableGradients.map(g => {
        if (!g.tensor) return null;
        return {
            name: g.name,
            values: g.tensor.dataSync(), // Grab numbers now
            shape: g.tensor.shape,
            size: g.tensor.size
        };
    });

    // 2. Now run your async loop using the saved 'values'
    for (const data of gradData) {
        if (!data) continue;

        const { name, values, shape, size } = data;
        const chunkSize = Math.ceil(size / this.numCores);

        // Pass the raw 'values' to your SAB manager
        const sabSet = this.getOrCreateSABs(name, values, shape, size);

        const promises = this.workers.map((worker, i) => {
            return new Promise((resolve) => {
                const start = i * chunkSize;
                const end = Math.min(start + chunkSize, size);

                worker.once('message', resolve);
                worker.postMessage({
                    range: [start, end],
                    buffers: sabSet,
                    params: {
                        lr: this.learningRate,
                        nextM_math: this.beta1,
                        nextV_math: this.beta2,
                        eps: this.epsilon
                    }
                });
            });
        });

        await Promise.all(promises);

        // 3. Sync back to TFJS
        const variable = this.registered_variables[name];
        variable.assign(tf.tensor(new Float32Array(sabSet.sharedVar), shape));
    }
}
    static get className() {
        return 'EvieOptimizer';
    }
}

// 💥 CRITICAL STEP: Registers it with TensorFlow.js
tf.serialization.registerClass(EvieOptimizer);

// Helper function to match your exact syntax: evie(0.01)
function Evie(lr) {
    return new EvieOptimizer(lr);
}

export { Evie }