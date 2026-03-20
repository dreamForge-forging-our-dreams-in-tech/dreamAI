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

app.get('/OS-information', async (req, res) => {
  try {
    const toGB = (bytes) => (bytes / (1024 ** 3)).toFixed(2);
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();

    // Initialize defaults in case hardware calls fail
    let cpuMainTemp = "N/A";
    let gpus = {};

    try {
      const cpuTemp = await si.cpuTemperature();
      cpuMainTemp = cpuTemp.main || "N/A";
    } catch (e) { console.error("CPU temp fail"); }

    try {
      const gpuData = await si.graphics();
      if (gpuData.controllers) {
        gpuData.controllers.forEach((gpu, index) => {
          gpus[index] = {
            "index": index,
            "model": gpu.model || "Unknown",
            "temp": gpu.temperatureGpu || "N/A"
          }
        });
      }
    } catch (e) { console.error("GPU data fail"); }

    res.json({
      "Total RAM": toGB(totalMemory),
      "Avaialble RAM": toGB(freeMemory),
      "Used RAM": toGB(totalMemory - freeMemory),
      "CPU Temp": cpuMainTemp,
      "GPUs": gpus,
      "Process": process.memoryUsage()
    });

  } catch (error) {
    console.error("Global OS Info Error:", error);
    // CRITICAL: Always send a JSON response even on failure
    res.status(500).json({ error: "Internal Server Error", GPUs: {} });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ dreamAi interface is active at http://localhost:${PORT}`);
});