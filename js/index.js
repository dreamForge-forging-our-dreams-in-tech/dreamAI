let location = window.location.origin; // location of the url bar used to send to correct api endpoints

let hardware_information = document.getElementById('display');

let training_progress = document.getElementById('training_progress');

async function sendChat() {
    checkStatuses(); // checks all statuses when a new message is send

    const box = document.getElementById('chatbox');
    const input = document.getElementById('userInput');
    const text = input.value;
    if (!text) return;

    // Display User Message
    box.innerHTML += `<p><span class="user">You:</span> ${text}</p>`;
    input.value = '';
    box.scrollTop = box.scrollHeight;

    try {
        const res = await fetch(location + '/chat', {
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
        box.scrollTop = box.scrollHeight;
    }
}

// make calls to all api ends and checks for the statusses they return and shows them to the user.
function checkStatuses() {
    fetch(location + '/OS-information')
        .then(res => res.json())
        .then(data => {
            hardware_information.innerText = `rss: ${Math.round(data.Process.rss / 1024 / 1024)} MB,          -- Resident Set Size
  heapTotal: ${Math.round(data.Process.heapTotal / 1024 / 1024)} MB, -- Total V8 heap
  heapUsed: ${Math.round(data.Process.heapUsed / 1024 / 1024)} MB,   -- Actual memory used
  external: ${Math.round(data.Process.external / 1024 / 1024)} MB,   -- C++ objects
  
  total ram: ${data["Total RAM"]}
  used ram: ${data["Used RAM"]}
  availalbe ram: ${data["Avaialble RAM"]}

  CPU temperature: ${data['CPU Temp']} °C

  `;

            let i;

            for (i of Object.keys(data.GPUs)) {
                hardware_information.innerText += `Index: ${i}
                GPU: ${data.GPUs[i].model}
                Temperature: ${data.GPUs[i].temp}`
            }
        });

    fetch(location + '/progress')
        .then(res => res.json())
        .then(data => {
            data = JSON.parse(data);

            if (data.total_epochs === 'undefined' || data.total_epochs == data.epoch) {
                training_progress.innerHTML = `Training finished/not in progress.<br>`;
            } else {
                training_progress.innerHTML = `Training in progress: <br>
                Epoch: ${data.epoch}/${data.total_epochs} <br>
                Loss: ${data.loss} <br>`;
            }

            training_progress.innerHTML += `
            Time passed: ${data.time_passed}<br>
                Started at: ${data.start_time}<br>
                Ended at: ${data.end_time}<br>
                `;
        });
}

window.setInterval(checkStatuses, 1000);

//connect events to elements
document.getElementById('sendButton').addEventListener('click', sendChat);

document.getElementById('userInput').addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        sendChat();
    }
});