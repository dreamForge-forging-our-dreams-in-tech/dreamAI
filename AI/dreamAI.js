import * as nodeUtil from 'util';
import { readFile } from 'node:fs/promises';

// 1. Import core and backends
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-cpu';

import { Evie } from './optimiser/evie.js';
let EvieOptimizerInstance = new Evie(0.02); // Create an instance of the Evie optimizer with a learning rate of 0.005

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

let total_epochs = 27; // how many epochs training will have.
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

    async loadCharacter(character_file_path) {
        if(this.character_JSON) { // character already loaded, just return the json.
            console.log('Character already loaded')
            return this.character_JSON;
        }

        let data = await readFile(character_file_path, 'utf8');
        let character_json = JSON.parse(data);

        this.character_JSON = character_json;
        return this.character_JSON;
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

    async compile_ai() {
        try {
            trainingData = await this.build_trainging_data();

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
                optimizer: EvieOptimizerInstance // Custom optimizer with a learning rate of 0.005
            });

            return { inputs, labels }
        } catch (error) {
            console.error('Error compiling AI:', error);
            throw error;
        }
    }

    async build_trainging_data() {
        let tokenized_data = [];
        let training_data_array = [
            // --- GREETINGS ---
            "User: hi. Response: Hello! How are you today? <END>",
            "User: hello. Response: Hi there! What is on your mind? <END>",
            "User: good morning. Response: Good morning! I hope you have a wonderful day. <END>",
            "User: greetings. Response: Hello! Pleased to meet you. <END>",

            // // --- IDENTITY / NAME ---
            `User: what is your name?. Response: My name is [${this.character_JSON.NAME}]. <END>`,
            `User: who are you?. Response: I'm ${this.character_JSON.NAME}, your friendly AI companion! <END>`,
            `User: tell me your name. Response: You can call me ${this.character_JSON.NAME}. <END>`,
            `User: what should i call you?. Response: Please call me ${this.character_JSON.NAME}! <END>`,

            // --- SMALL TALK & WELL BEING ---
            "User: how are you?. Response: I am doing great, thank you for asking! How are you? <END>",
            "User: how is it going?. Response: It is going very well. What about you? <END>",
            "User: are you okay?. Response: Yes, I am functioning perfectly! <END>",
            "User: what are you doing?. Response: Just hanging out here, ready to chat with you. <END>",

            // --- HUMAN INTERACTION REACTION ---
            "User: i am sad. Response: Oh no, I am sorry to hear that. Can I help cheer you up? <END>",
            "User: i am happy today. Response: That is wonderful! What made your day so good? <END>",
            "User: i am bored. Response: Let us play a game or talk about something interesting then! <END>",
            "User: thank you. Response: You are very welcome! <END>",
            "User: thanks marie. Response: No problem at all, happy to help! <END>",

            // --- CLOSINGS / GOODBYES ---
            "User: bye. Response: Goodbye! Have a great day! <END>",
            "User: see you later. Response: See you later! Take care. <END>",
            "User: goodnight. Response: Goodnight! Sleep well. <END>",
            "User: i have to go. Response: Alright, talk to you next time! <END>"
        ];

        let maxIdFound = 0; // Keep track of the biggest number

        tokenizer.prepare_data(training_data_array); // prepare dataset for tokenization and add new words to tokenizer_data.json if needed.
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

    async train() {
        // Switch to CPU for training to avoid "Kernel not registered" errors

        await setTrainingBackend();
        const { inputs, labels } = await this.compile_ai();

        const xs = tf.tensor2d(inputs, [inputs.length, seqLength], 'int32');
        const ys = tf.tensor1d(labels, 'float32');

        console.log("🚀 Training starting on CPU Backend...");
        const start = new Date();
        let epoch_start;

        await model.fit(xs, ys, {
            epochs: total_epochs,
            callbacks: {
                onEpochBegin: (epoch, logs) => {
                    epoch_start = new Date();
                },

                onEpochEnd: (epoch, logs) => {
                    const end = new Date();

                    // Calculate the difference in total milliseconds
                    const diff = end - start;
                    const epoch_diff = end - epoch_start;

                    const mins = Math.floor(diff / 60000);
                    const secs = Math.floor((diff % 60000) / 1000);
                    const mms = diff % 1000;

                    const epoch_mins = Math.floor(epoch_diff / 60000);
                    const epoch_secs = Math.floor((epoch_diff % 60000) / 1000);
                    const epoch_mms = epoch_diff % 1000;

                    training_progress = {
                        epoch: epoch + 1,
                        total_epochs: total_epochs,

                        learningRate: EvieOptimizerInstance.learningRate,

                        loss: logs.loss.toFixed(4),
                        lossDiff: epoch === 0 ? 0 : (training_progress.loss - logs.loss).toFixed(4),

                        overshot: epoch > 0 ? (training_progress.loss - logs.loss) < 0 : false,

                        time_passed: `${mins}:${secs}:${Math.floor(mms / 100)}`, // Format as MM:SS.MS
                        epoch_time: `${epoch_mins}:${epoch_secs}:${Math.floor(epoch_mms / 100)}`,
                        start_time: start.toString(),
                        end_time: end.toString()
                    };

                    console.log(training_progress);
                }
            }
        });

        xs.dispose();
        ys.dispose();

        console.log("✅ Training complete.");
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