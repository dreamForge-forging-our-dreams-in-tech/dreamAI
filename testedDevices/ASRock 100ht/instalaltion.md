# Installation TrueNAS ASRock 100ht

First cd into the folder where dreamAI is located in the shell!

Installation command: sudo docker run -it --rm --user 0:0 --network=host -v "$(pwd):/app" -v dreamai_modules:/app/node_modules -w /app node:20-bookworm bash -c "npm install && node server.js"