// index.js
import brain from 'https://esm.sh/brain.js';

const net = new brain.NeuralNetwork();

net.train([{ input: [0, 0], output: [0] }, { input: [1, 1], output: [1] }]);

const output = net.run([1, 0]); 
console.log(output);