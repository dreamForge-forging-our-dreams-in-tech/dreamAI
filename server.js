import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

import os from 'os';
import si from 'systeminformation';

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

    dream_ai.train('./characters/Marie.json').then(async () => {
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

app.get('/OS-information', async (re, res) => { // returns all the available debugging information for the OS
  try {
    const toGB = (bytes) => (bytes / (1024 ** 3)).toFixed(2);

    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    const usedMemory = totalMemory - freeMemory;

    // 1. CPU Temperature
    const cpuTemp = await si.cpuTemperature();

    // 2. GPU Data (includes temperature where supported)
    let gpu_data_array = {};
    const gpuData = await si.graphics();
    gpuData.controllers.forEach((gpu, index) => {
      gpu_data_array[index] = {
        "index": index,
        "model": gpu.model,
        "temp": gpu.temperatureGpu
      }
    });

    res.json({
      "Total RAM": toGB(totalMemory),
      "Avaialble RAM": toGB(freeMemory),
      "Used RAM": toGB(usedMemory),
      "CPU Temp": cpuTemp.main,
      "GPUs": gpu_data_array,
      "Process": process.memoryUsage()
    });

  } catch (e) {
    console.error("Error fetching temps:", e);
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ dreamAi interface is active at http://localhost:${PORT}`);
});