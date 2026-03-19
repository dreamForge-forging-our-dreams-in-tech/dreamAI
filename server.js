import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

import { dreamAI } from './AI/dreamAI.js';
let dream_ai = new dreamAI();

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

    dream_ai.train().then(async () => {
      const reply = await dream_ai.generate_prompt("Hero", userPrompt);
      console.log("\nAI says:", reply);

      res.json({ reply: reply });
    });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Check if Ollama is running on the ASRock." });
  }
});

app.get('/progress', async (req, res) => { // trianing progress api endpoint, returns training progress
  res.json(dream_ai.get_training_progress());
});

app.get('/memory', async (re, res) => { // memory api endpoint returns all used memory by the AI and node.
  res.json(process.memoryUsage());
});

app.listen(PORT, () => {
  console.log(`✅ dreamAi interface is active at http://localhost:${PORT}`);
});