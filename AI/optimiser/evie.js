import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-cpu';

import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import path from 'path';

import os from 'os'

// Fix for __dirname in ESM
const __dirname = path.dirname(fileURLToPath(import.meta.url));

console.log(os.availableParallelism(), "CPU cores detected, using", os.availableParallelism() - 2, "for training.");
class EvieOptimizer extends tf.Optimizer {
    constructor(learningRate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        super();
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.accumulators = new Map();
        this.numCores = 2;
        this.workers = [];

        // Use a private map to store the actual Variable references
        this.variableMap = new Map();

        for (let i = 0; i < this.numCores; i++) {
            console.log(`Spawning worker ${i + 1}/${this.numCores}`);

            const workerPath = path.join(__dirname, 'optimiser_worker.js');
            this.workers.push(new Worker(workerPath));
        }
    }

    async applyGradients(variableGradients) {
        // 1. THE EMERGENCY RESCUE (Synchronous)
        // No 'await' allowed before this block!
        const snapshots = [];

        const gradients = Array.isArray(variableGradients)
            ? variableGradients
            : Object.keys(variableGradients).map(name => ({ name, tensor: variableGradients[name] }));

        for (const g of gradients) {
            if (!g.tensor) continue;

            // Find the actual weight variable to get the current weight values
            const variable = tf.engine().registeredVariables[g.name] || this.variableMap.get(g.name);

            if (!this.variableMap.has(g.name)) {
                this.variableMap.set(g.name, g.tensor);
            }

            // SNAPSHOT BOTH: The Gradient AND the Current Weight
            snapshots.push({
                name: g.name,
                gradValues: g.tensor.dataSync(),      // Rescue the Gradient
                currentWeights: variable ? variable.dataSync() : null, // Rescue the Weight
                shape: g.tensor.shape,
                size: g.tensor.size
            });
        }

        // 2. THE WORKER PHASE (Asynchronous)
        for (const data of snapshots) {
            const { name, gradValues, currentWeights, shape, size } = data;
            const chunkSize = Math.ceil(size / this.numCores);

            // Pass the pre-rescued 'gradValues' and 'currentWeights'
            const sabSet = this.getOrCreateSABs(name, gradValues, currentWeights, shape, size);

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

            // 3. APPLY BACK
            const variable = tf.engine().registeredVariables[name] || this.variableMap.get(name);
            if (variable && variable.assign) {
                tf.tidy(() => {
                    variable.assign(tf.tensor(new Float32Array(sabSet.sharedVar), shape));
                });
            }
        }
    }

    getOrCreateSABs(name, rescuedGrad, rescuedWeights, shape, size) {
        if (this.accumulators.has(name)) {
            const sabs = this.accumulators.get(name);
            new Float32Array(sabs.sharedGrad).set(rescuedGrad);
            return sabs;
        }

        const byteLength = size * 4;
        const sabs = {
            sharedGrad: new SharedArrayBuffer(byteLength),
            sharedM: new SharedArrayBuffer(byteLength),
            sharedV: new SharedArrayBuffer(byteLength),
            sharedVar: new SharedArrayBuffer(byteLength)
        };

        // Use the values we rescued at the top of applyGradients
        if (rescuedWeights) {
            new Float32Array(sabs.sharedVar).set(rescuedWeights);
        }

        new Float32Array(sabs.sharedM).fill(0);
        new Float32Array(sabs.sharedV).fill(0);
        new Float32Array(sabs.sharedGrad).set(rescuedGrad);

        this.accumulators.set(name, sabs);
        return sabs;
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