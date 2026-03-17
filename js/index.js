async function sendChat() {
    const box = document.getElementById('chatbox');
    const input = document.getElementById('userInput');
    const text = input.value;
    if (!text) return;

    // Display User Message
    box.innerHTML += `<p><span class="user">You:</span> ${text}</p>`;
    input.value = '';

    try {
        const res = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: text })
        });
        const data = await res.json();

        // Display AI Message
        box.innerHTML += `<p><span class="ai">Dolphin:</span> ${data.reply}</p>`;
        box.scrollTop = box.scrollHeight;
    } catch (e) {
        box.innerHTML += `<p style="color:red">Error connecting to your JS server.</p>`;
    }
}

function checkMemory() {
    fetch('http://localhost:5000/memory')
        .then(res => res.json())
        .then(data => {
            document.getElementById('display').innerText = `rss: ${Math.round(data.rss / 1024 / 1024)} MB,          -- Resident Set Size
  heapTotal: ${Math.round(data.heapTotal / 1024 / 1024)} MB, -- Total V8 heap
  heapUsed: ${Math.round(data.heapUsed / 1024 / 1024)} MB,   -- Actual memory used
  external: ${Math.round(data.external / 1024 / 1024)} MB,   -- C++ objects`;
        });
}

window.setInterval(checkMemory, 1);

document.getElementById('sendButton').addEventListener('click', sendChat);

document.getElementById('userInput').addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        sendChat();
    }
});