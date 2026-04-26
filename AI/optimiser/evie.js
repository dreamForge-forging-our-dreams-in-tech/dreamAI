import * as tf from '@tensorflow/tfjs';
import Piscina from 'piscina';
import { fileURLToPath } from 'url';
import path from 'path';
import os from 'os';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

class EvieOptimizer extends tf.Optimizer {
    constructor(learningRate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        super();
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.accumulators = new Map();
        this.variableMap = new Map();

        this.threads = Math.floor(os.availableParallelism() / 100 * 75); // Use 75% of available cpu cores

        // Initialize Piscina Pool once
        this.pool = new Piscina({
            filename: path.resolve(__dirname, 'optimiser_worker.cjs'),

            resourceLimits: {
                maxOldGenerationSizeMb: 512 // Prevents GC from ballooning
            },

            maxQueue: 30, // Hard limit
            maxThreads: this.threads, // Leave 1 core for the Main Thread, Lock threads to physical cores for stability
            minThreads: this.threads, // Prevent "Cold Starts" of threads
            idleTimeout: 30000, // Keep them alive  
            concurrentTasksPerWorker: 1, // Forces linear math, reducing fluctuation

            waitOnQueue: true, // IMPORTANT: This makes pool.run return a promise that waits
        });
    }

    async applyGradients(variableGradients) {

        const snapshots = [];
        const gradients = Array.isArray(variableGradients)
            ? variableGradients
            : Object.keys(variableGradients).map(name => ({ name, tensor: variableGradients[name] }));

        for (const g of gradients) {
            if (!g.tensor) continue;

            const variable = tf.engine().registeredVariables[g.name] || this.variableMap.get(g.name);
            if (!this.variableMap.has(g.name)) {
                this.variableMap.set(g.name, g.tensor);
            }

            // Sync rescue to avoid the "Tensor disposed" error
            snapshots.push({
                name: g.name,
                gradValues: g.tensor.dataSync(),
                currentWeights: variable ? variable.dataSync() : null,
                shape: g.tensor.shape,
                size: g.tensor.size
            });
        }

        // HARD THROTTLE: If workers are busy, wait.

        // This prevents the 'TaskQueueAtLimit' crash on 4-core devices.

        while (this.pool.queueSize > 0) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }

        // Processing Phase
        for (const data of snapshots) {
            const { name, gradValues, currentWeights, shape, size } = data;
            const numThreads = this.pool.options.maxThreads;
            const chunkSize = Math.ceil(size / numThreads);

            const sabSet = this.getOrCreateSABs(name, gradValues, currentWeights, shape, size);

            const tasks = [];
            for (let i = 0; i < numThreads; i++) {
                const start = i * chunkSize;
                const end = Math.min(start + chunkSize, size);

                if (start >= size) break;

                // piscina.run() replaces worker.postMessage()
                tasks.push(this.pool.run({
                    range: [start, end],
                    buffers: sabSet,
                    params: {
                        lr: this.learningRate,
                        nextM_math: this.beta1,
                        nextV_math: this.beta2,
                        eps: this.epsilon
                    }
                }));
            }

            // Wait for all chunks of this specific variable to finish
            await Promise.all(tasks);

            const variable = tf.engine().registeredVariables[name] || this.variableMap.get(name);
            if (variable && variable.assign) {
                // Create the new tensor
                const nextWeights = tf.tensor(new Float32Array(sabSet.sharedVar), shape);
                variable.assign(nextWeights);
                // MANUALLY DISPOSE the temporary tensor immediately
                nextWeights.dispose();
            }
        }

        await new Promise(resolve => setImmediate(resolve));
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

        if (rescuedWeights) new Float32Array(sabs.sharedVar).set(rescuedWeights);
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