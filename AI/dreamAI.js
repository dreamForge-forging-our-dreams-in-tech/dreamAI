import * as nodeUtil from 'util';
import { readFile } from 'node:fs/promises';

// 1. Import core and backends
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-cpu';

import { Evie } from './optimiser/evie.js';

import { Tokenizer } from './tokenizer/tokenizer.js';
let tokenizer = new Tokenizer();

import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import path from 'path';

// Fix for __dirname in ESM
const __dirname = path.dirname(fileURLToPath(import.meta.url));

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

let total_epochs = 50; // how many epochs training will have.
let ai_padding_token = 0; //what the padding token will be for the ai

let training_progress = {};
let model;
let trainingData = [];
let vocabSize = 0;
let seqLength = 10;
let embeddingDims;


class dreamAI {
    constructor() {
    }

    get_training_progress() {
        return JSON.stringify(training_progress);
    }

    prepareDataset(tokenizedSentences, maxSequenceLen) {
        let inputs = [];
        let labels = [];

        // 1. Loop through each sentence in your master array
        for (let sentence of tokenizedSentences) {

            // 2. For each sentence, slide across it word by word
            for (let i = 1; i < sentence.length; i++) {
                let inputSequence = sentence.slice(0, i); // Words up to now
                const nextWord = sentence[i];             // The single target word

                // 3. Pad the front of the sequence with zeros so it's exactly maxSequenceLen
                while (inputSequence.length < maxSequenceLen) {
                    inputSequence.unshift(ai_padding_token); // Add padding token (0) to the front
                }

                // 4. If it's too long, trim the oldest words from the front
                if (inputSequence.length > maxSequenceLen) {
                    inputSequence = inputSequence.slice(inputSequence.length - maxSequenceLen);
                }

                inputs.push(inputSequence);
                labels.push(nextWord);
            }
        }

        return { inputs, labels };

    }

    async compile_ai(character_file_path) {
        try {
            trainingData = await this.build_trainging_data(character_file_path);

            const { inputs, labels } = this.prepareDataset(trainingData, seqLength);

            embeddingDims = 32; // How many numbers describe a single word

            model = tf.sequential();

            // 1. Embedding Layer: Converts letter IDs into 16-dimensional vectors
            model.add(tf.layers.embedding({
                inputDim: vocabSize,
                outputDim: embeddingDims,
                inputLength: seqLength // e.g., 10
            }));

            // 2. LSTM layer: Remembers the order of the letters
            model.add(tf.layers.lstm({ units: 32, returnSequences: false })); // Bumped to 128 for more memory

            // 3. Dense layer: Gives a probability score for every character in your vocab
            model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

            model.compile({
                loss: 'sparseCategoricalCrossentropy', // 🔥 Fixed for integer targets!
                optimizer: Evie(0.005) // Custom optimizer with a learning rate of 0.005
            });

            return { inputs, labels }
        } catch (error) {
            console.error('Error compiling AI:', error);
            throw error;
        }
    }

    async build_trainging_data(character_file_path) {
        const data = await readFile(character_file_path, 'utf8');
        const character_json = JSON.parse(data);

        let tokenized_data = [];
        let training_data_array = [
            // --- GREETINGS ---
            "Context: . User: hi. Response: Hello! How are you today? <END>",
            "Context: . User: hello. Response: Hi there! What is on your mind? <END>",
            "Context: . User: good morning. Response: Good morning! I hope you have a wonderful day. <END>",
            "Context: . User: greetings. Response: Hello! Pleased to meet you. <END>",

            // // --- IDENTITY / NAME ---
            `Context: . User: what is your name?. Response: My name is ${character_json.Name}. <END>`,
            `Context: . User: who are you?. Response: I am ${character_json.Name}, your friendly AI companion! <END>`,
            `Context: . User: tell me your name. Response: You can call me ${character_json.Name}. <END>`,
            `Context: . User: what should i call you?. Response: Please call me ${character_json.Name}! <END>`,

            // --- SMALL TALK & WELL BEING ---
            "Context: . User: how are you?. Response: I am doing great, thank you for asking! How are you? <END>",
            "Context: . User: how is it going?. Response: It is going very well. What about you? <END>",
            "Context: . User: are you okay?. Response: Yes, I am functioning perfectly! <END>",
            "Context: . User: what are you doing?. Response: Just hanging out here, ready to chat with you. <END>",

            // --- HUMAN INTERACTION REACTION ---
            "Context: . User: i am sad. Response: Oh no, I am sorry to hear that. Can I help cheer you up? <END>",
            "Context: . User: i am happy today. Response: That is wonderful! What made your day so good? <END>",
            "Context: . User: i am bored. Response: Let us play a game or talk about something interesting then! <END>",
            "Context: . User: thank you. Response: You are very welcome! <END>",
            "Context: . User: thanks marie. Response: No problem at all, happy to help! <END>",

            // --- CLOSINGS / GOODBYES ---
            "Context: . User: bye. Response: Goodbye! Have a great day! <END>",
            "Context: . User: see you later. Response: See you later! Take care. <END>",
            "Context: . User: goodnight. Response: Goodnight! Sleep well. <END>",
            "Context: . User: i have to go. Response: Alright, talk to you next time! <END>"
        ];

        let maxIdFound = 0; // Keep track of the biggest number

        for (let i of training_data_array) {
            const tokens = tokenizer.tokenize(i);
            tokenized_data.push(tokens);

            // 🔍 Find the biggest ID inside this specific token array
            const biggestInSentence = Math.max(...tokens);

            // Update our master tracker if this sentence has a bigger number!
            if (biggestInSentence > maxIdFound) {
                maxIdFound = biggestInSentence;
            }
        }

        // ✅ Set vocabSize to the highest ID + 1 (to account for zero padding!)
        vocabSize = maxIdFound + 10;
        vocabSize = Math.ceil(vocabSize);

        return tokenized_data;
    }

    async train(character_file_path) {
        // Switch to CPU for training to avoid "Kernel not registered" errors

        await setTrainingBackend();
        const { inputs, labels } = await this.compile_ai(character_file_path);

        const xs = tf.tensor2d(inputs);
        const ys = tf.tensor2d(labels.map(l => [l]), [labels.length, 1], 'float32');

        console.log("🚀 Training starting on CPU Backend...");
        console.time();
        await model.fit(xs, ys, {
            epochs: total_epochs,
            callbacks: {
                onBatchEnd: async (batch, logs) => {
                    // if (batch % 20 === 0) {
                    //     await new Promise(resolve => setImmediate(resolve));
                    // }
                },
                onEpochEnd: (epoch, logs) => {
                    training_progress = {
                        epoch: epoch + 1,
                        loss: logs.loss.toFixed(4),
                        total_epochs: total_epochs
                    };
                    console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
                }
            }
        });

        xs.dispose();
        ys.dispose();

        console.log("✅ Training complete.");
        console.timeEnd();
    }

    async generate_prompt(seedWords, maxLenToGenerate = 30) {
        let generatedWords = [...seedWords]; // Start with your user prompt tokens
        let response = []; // where the actual response of the ai wil be stored without the user message in it.

        for (let i = 0; i < maxLenToGenerate; i++) {
            // 1. Pad the current input sequence to match the seqLength your model expects
            let inputSequence = [...generatedWords];

            while (inputSequence.length < seqLength) {
                inputSequence.unshift(0); // Pad front with 0s
            }
            if (inputSequence.length > seqLength) {
                inputSequence = inputSequence.slice(-seqLength); // Trim if too long
            }

            const nextWordId = tf.tidy(() => {
                const inputTensor = tf.tensor2d([inputSequence], [1, seqLength]);
                const logits = model.predict(inputTensor); // [1, vocabSize]

                // 🎯 No temperature, no softmax, no multinomial! Just grab the highest peak!
                const bestWordTensor = logits.argMax(1);
                return bestWordTensor.dataSync()[0];
            });

            // 3. Add predicted word to our running sequence
            generatedWords.push(nextWordId);
            response.push(nextWordId);

            // Optional: Break early if the AI outputs an <END> token (if you have one)
            if (nextWordId === 0) {
                break;
            }
        }

        console.log(response.join().replaceAll(',', ' '))

        return response.join().replaceAll(',', ' ');
    }
}

export { dreamAI };