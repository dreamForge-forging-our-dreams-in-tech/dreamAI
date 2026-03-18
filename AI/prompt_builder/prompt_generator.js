import * as tf from '@tensorflow/tfjs-node';
import { TinyLLM } from './model.js';
import fs from 'fs';
import path from 'path';
import { Tokenizer } from '../tokenizer/tokenizer.js';

let tokenizer = new Tokenizer();
const vocabSize = 5000;
const embeddingDim = 64;
const myAI = new TinyLLM(vocabSize, embeddingDim);

// Compile the model once at the top level
myAI.model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy'
});

export class dreamAI {
    constructor() {
        this.maxLen = 10; // The sequence length your model expects
    }

    async trainFromFile(filePath) {
        return new Promise(async (resolve, reject) => {
            try {
                const rawData = JSON.parse(fs.readFileSync(path.join(process.cwd(), filePath), 'utf8'));
                console.log(`Starting training on ${rawData.length} samples...`);

                for (let sentence of rawData) {
                    const tokens = tokenizer.tokenize(sentence);

                    // Sliding window training: Learning the sequence word by word
                    for (let i = 1; i < tokens.length; i++) {
                        const inputSeq = tokens.slice(0, i);
                        const targetId = tokens[i];

                        // Prepare Padded Input
                        const padded = inputSeq.concat(Array(this.maxLen - inputSeq.length).fill(0)).slice(0, this.maxLen);

                        const xs = tf.tensor2d([padded]);
                        const ys = tf.oneHot(tf.tensor1d([targetId], 'int32'), vocabSize);

                        // model.fit is already a promise, so we await it
                        await myAI.model.fit(xs, ys, {
                            epochs: 3,
                            verbose: 0
                        });

                        xs.dispose();
                        ys.dispose();
                    }
                }
                console.log("Training Sequence Complete.");
                resolve(true); // This tells 'await' that we are officially done
            } catch (error) {
                console.error("Training failed:", error);
                reject(error);
            }
        });
    }

    async generate_prompt(seedText, wordCount = 5) {
        // 1. Initial conversion of your prompt to numbers
        let currentTokens = Array.from(tokenizer.tokenize(seedText));
        let fullSentence = seedText;

        for (let i = 0; i < wordCount; i++) {
            // Prepare the 10-slot input for the ASRock CPU
            const paddedInput = Array.from(currentTokens.concat(Array(this.maxLen - currentTokens.length).fill(0)).slice(0, this.maxLen));
            const inputTensor = tf.tensor2d(paddedInput, [1, this.maxLen]);

            // 2. AI predicts the ID of the next word
            const prediction = myAI.model.predict(inputTensor);
            const nextId = prediction.argMax(-1).dataSync()[0];

            // 3. YOUR de_tokenizer turns that ID into a word string
            const nextWord = tokenizer.de_tokenize(nextId);

            // 4. Update the sentence string for the user
            fullSentence += " " + nextWord;

            // 5. CRITICAL: Add the ID back to the tokens list for the next loop
            currentTokens.push(nextId);

            // Cleanup RAM
            inputTensor.dispose();
            prediction.dispose();

            if (nextWord === "?" || nextWord === ".") break;
        }

        return fullSentence;
    }
}