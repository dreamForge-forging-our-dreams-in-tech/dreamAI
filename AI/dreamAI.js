if (typeof nodeUtil.isNullOrUndefined !== 'function') {
    nodeUtil.isNullOrUndefined = (v) => v === null || v === undefined;
}

import * as tf from '@tensorflow/tfjs-node';
import * as nodeUtil from 'util';

let training_progress = {};

// --- DATA & RAG CONFIG ---
const trainingData = [
    "Context: Hero. User: Hi. Response: Greetings!",
    "Context: Hero. User: Help. Response: I shall protect you.",
    "Context: Villain. User: Hi. Response: Foolish mortal.",
    "Context: Villain. User: Stop. Response: Never! The world is mine."
];

const text = trainingData.join('\n');
const chars = [...new Set(text)].sort();
const charToId = Object.fromEntries(chars.map((c, i) => [c, i]));
const idToChar = Object.fromEntries(chars.map((c, i) => [i, c]));
const vocabSize = chars.length;
const seqLength = 30; // Matches your ASRock's limited cache better

// --- MODEL ARCHITECTURE ---
const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 16, inputLength: seqLength }));
model.add(tf.layers.lstm({ units: 64, returnSequences: false })); // Lower units for speed
model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(0.01) // Slightly higher learning rate for small data
});

class dreamAI {
    construct() {
    }

    get_training_progress() { // returns the training progress
        return JSON.stringify(training_progress);
    }

    // --- TRAINING ---
    async train(characterDefinition) { // train the ai on the available training generated based on the characters definition
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