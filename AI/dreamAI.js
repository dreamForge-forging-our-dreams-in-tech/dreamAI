import * as nodeUtil from 'util';
import { readFile } from 'node:fs/promises';

// 1. Import core and backends
import * as tf from '@tensorflow/tfjs'; 
import '@tensorflow/tfjs-backend-wasm'; 
import '@tensorflow/tfjs-backend-cpu'; 

// Polyfill for older environments
if (typeof nodeUtil.isNullOrUndefined !== 'function') {
    nodeUtil.isNullOrUndefined = (v) => v === null || v === undefined;
}

/**
 * Strategy: We use CPU for training because WASM lacks 'UnsortedSegmentSum'.
 * We use WASM for generation (inference) because it is faster.
 */
async function setTrainingBackend() {
    await tf.setBackend('cpu');
    await tf.ready();
    console.log("⚙️ Training Backend set to:", tf.getBackend());
}

async function setInferenceBackend() {
    try {
        await tf.setBackend('wasm');
        await tf.ready();
        console.log("⚡ Inference Backend set to:", tf.getBackend());
    } catch (e) {
        await tf.setBackend('cpu');
    }
}

let training_progress = {};
let model;
let trainingData = [];
let text, chars, charToId, idToChar, vocabSize;
let seqLength = 30; 

class dreamAI {
    constructor() {}

    get_training_progress() {
        return JSON.stringify(training_progress);
    }

    async compile_ai(character_file_path) {
        try {
            trainingData = await this.build_trainging_data(character_file_path);
            text = trainingData.join('\n');
            chars = [...new Set(text)].sort();
            charToId = Object.fromEntries(chars.map((c, i) => [c, i]));
            idToChar = Object.fromEntries(chars.map((c, i) => [i, c]));
            vocabSize = chars.length;

            model = tf.sequential();
            model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 16, inputLength: seqLength }));
            model.add(tf.layers.lstm({ units: 64, returnSequences: false })); 
            model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

            model.compile({
                loss: 'categoricalCrossentropy',
                optimizer: tf.train.adam(0.01)
            });
        } catch (error) {
            console.error('Error compiling AI:', error);
            throw error;
        }
    }

    async build_trainging_data(character_file_path) {
        const data = await readFile(character_file_path, 'utf8');
        const character_json = JSON.parse(data);
        return [
            `Context: . User: What is your name?. Response: My name is ${character_json.Name}`,
            `Context: . User: Hello ${character_json.Name}. Response: Hi, how are you?`,
        ];
    }

    async train(character_file_path) {
        // Switch to CPU for training to avoid "Kernel not registered" errors
        await setTrainingBackend();
        await this.compile_ai(character_file_path);

        const inputs = [];
        const labels = [];
        for (let i = 0; i < text.length - seqLength; i++) {
            inputs.push(text.slice(i, i + seqLength).split('').map(c => charToId[c]));
            labels.push(charToId[text[i + seqLength]]);
        }

        const xs = tf.tensor2d(inputs);
        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), vocabSize);

        console.log("🚀 Training starting on CPU Backend...");
        await model.fit(xs, ys, {
            epochs: 100,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    training_progress = {
                        epoch: epoch + 1,
                        loss: logs.loss.toFixed(4),
                        total_epochs: 100
                    };
                    console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
                }
            }
        });
        
        xs.dispose();
        ys.dispose();
        console.log("✅ Training complete.");
    }

    async generate_prompt(context, userMsg) {
        // Switch to WASM for faster generation if possible
        await setInferenceBackend();
        
        let inputStr = `Context: ${context}. User: ${userMsg}. Response: `;
        let currentSeq = inputStr.slice(-seqLength).padStart(seqLength, ' ');
        let result = "";

        for (let i = 0; i < 40; i++) {
            const char = tf.tidy(() => {
                const inputTensor = tf.tensor2d([currentSeq.split('').map(c => charToId[c] || 0)]);
                const pred = model.predict(inputTensor);
                const id = tf.argMax(pred, 1).dataSync()[0];
                return idToChar[id];
            });

            if (char === '.' || char === '\n') break;
            result += char;
            currentSeq = (currentSeq + char).slice(-seqLength);
        }
        return result;
    }
}

export { dreamAI };