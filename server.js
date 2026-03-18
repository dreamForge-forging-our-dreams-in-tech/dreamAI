import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

import { Tokenizer } from './AI/tokenizer/tokenizer.js';
let tokenizer = new Tokenizer();

const app = express();
const PORT = 5000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- MIDDLEWARE ---
app.use(cors());
app.use(express.json());

// Serve static files from the current directory
// This makes http://localhost:5000/ look for index.html automatically
app.use(express.static(__dirname));

// OPTIONAL: Explicitly send index.html if the above fails
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// --- CHAT ENDPOINT ---
app.post('/chat', async (req, res) => {
     try {

    const userPrompt = req.body.prompt;

    const response = tokenizer.de_tokenize(tokenizer.tokenize(userPrompt));

    res.json({ reply: response });

  } catch (error) {
    console.error("Ollama Error:", error);
    res.status(500).json({ error: "Check if Ollama is running on the ASRock." });
  }
});

app.get('/memory', async (re, res) => {
    res.json(process.memoryUsage());
});

app.listen(PORT, () => {
  console.log(`✅ dreamAi interface is active at http://localhost:${PORT}`);
});