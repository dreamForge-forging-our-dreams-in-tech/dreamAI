if (typeof nodeUtil.isNullOrUndefined !== 'function') {
    nodeUtil.isNullOrUndefined = (v) => v === null || v === undefined;
}

import * as tf from '@tensorflow/tfjs-node';
import * as nodeUtil from 'util';
import { readFile } from 'node:fs/promises';

let training_progress = {};

let model;

// --- DATA & RAG CONFIG ---
// keep it empty as it will be filled
let trainingData = [];

let text;
let chars;
let charToId;
let idToChar;
let vocabSize;

let seqLength = 30; // Matches your ASRock's limited cache better

class dreamAI {
    construct() {
    }

    get_training_progress() { // returns the training progress
        return JSON.stringify(training_progress);
    }

    async compile_ai(character_file_path) { // compiel the ai using the training data and other variables.
        return new Promise(async (resolve, reject) => {
            try {

                trainingData = await this.build_trainging_data(character_file_path);

                text = trainingData.join('\n');
                chars = [...new Set(text)].sort();
                charToId = Object.fromEntries(chars.map((c, i) => [c, i]));
                idToChar = Object.fromEntries(chars.map((c, i) => [i, c]));
                vocabSize = chars.length;
                seqLength = 30;

                // --- MODEL ARCHITECTURE ---
                model = tf.sequential();
                model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 16, inputLength: seqLength }));
                model.add(tf.layers.lstm({ units: 64, returnSequences: false })); // Lower units for speed
                model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

                model.compile({
                    loss: 'categoricalCrossentropy',
                    optimizer: tf.train.adam(0.01) // Slightly higher learning rate for small data
                });
                resolve();
            } catch (error) {
                console.error('Error reading the JSON file:', error);
                reject(error);
            }
        });
    }

    async build_trainging_data(character_file_path) { // dynamically builds the trianing data of the AI using the character card json file
        return new Promise(async (resolve, reject) => {
            try {
                // Read the file as a string
                const data = await readFile(character_file_path, 'utf8');

                // Parse the string into a JavaScript object
                const character_json = JSON.parse(data);

                let training_data = [
                    `Context: . User: What is your name?. Response: My name is ${character_json.Name}`,
                    `Context: . User: Hello ${character_json.Name}. Response: Hi, how are you?`,
                ];
                resolve(training_data);
            } catch (error) {
                console.error('Error reading the JSON file:', error);
                reject(error);
            }
        });
    }

    // --- TRAINING ---
    async train(character_file_path) { // train the ai on the available training generated based on the characters definition
        await this.compile_ai(character_file_path);

        const inputs = [];
        const labels = [];
        for (let i = 0; i < text.length - seqLength; i++) {
            inputs.push(text.slice(i, i + seqLength).split('').map(c => charToId[c]));
            labels.push(charToId[text[i + seqLength]]);
        }

        const xs = tf.tensor2d(inputs);
        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), vocabSize);

        console.log("Training starting on C++ Backend...");
        await model.fit(xs, ys, {
            epochs: 100,
            callbacks: {
                onEpochEnd: function (epoch, logs) {
                    training_progress = {
                        epoch: epoch + 1,
                        loss: logs.loss.toFixed(4),
                        total_epochs: this.params.epochs
                    };

                    console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
                }
            }
        });
        console.log("Training complete.");
    }

    // --- GENERATION (The Roleplay Output) ---
    async generate_prompt(context, userMsg) { // generate a prompt to return to the user using the users message and rag context keywords
        let inputStr = `Context: ${context}. User: ${userMsg}. Response: `;
        let currentSeq = inputStr.slice(-seqLength).padStart(seqLength, ' ');
        let result = "";

        for (let i = 0; i < 40; i++) {
            const inputTensor = tf.tensor2d([currentSeq.split('').map(c => charToId[c] || 0)]);
            const pred = model.predict(inputTensor);
            const id = tf.argMax(pred, 1).dataSync()[0];
            const char = idToChar[id];

            if (char === '.' || char === '\n') break;
            result += char;
            currentSeq = (currentSeq + char).slice(-seqLength);
        }
        return result;
    }

}

export { dreamAI }