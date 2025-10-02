const API_URL = "http://127.0.0.1:8000/chat";

const $msgs  = document.getElementById("messages");
const $form  = document.getElementById("form");
const $input = document.getElementById("input");
const $send  = document.getElementById("send");

// Gespreksgeschiedenis (we sturen dit mee naar de backend)
let history = JSON.parse(localStorage.getItem("meesman_history") || "[]");

function saveHistory() {
  localStorage.setItem("meesman_history", JSON.stringify(history));
}

function renderMessage(role, content) {
  const row = document.createElement("div");
  row.className = `row ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  row.appendChild(bubble);
  $msgs.appendChild(row);
  $msgs.scrollTop = $msgs.scrollHeight;
}

function renderTyping(on) {
  let el = document.getElementById("typing");
  if (on) {
    if (!el) {
      el = document.createElement("div");
      el.id = "typing";
      el.className = "typing";
      el.textContent = "Assistant is aan het typen…";
      $msgs.appendChild(el);
      $msgs.scrollTop = $msgs.scrollHeight;
    }
  } else if (el) {
    el.remove();
  }
}

function renderFromHistory() {
  $msgs.innerHTML = "";
  history.forEach(m => renderMessage(m.role, m.content));
}

renderFromHistory();

// Event: form submit
$form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = ($input.value || "").trim();
  if (!text) return;

  // UI update
  renderMessage("user", text);
  $input.value = "";
  $input.focus();
  $send.disabled = true;
  renderTyping(true);

  // Update history (user)
  history.push({ role: "user", content: text });
  saveHistory();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        // de meeste backends willen het actuele bericht apart in `messages`
        messages: [{ role: "user", content: text }],
        // plus volledige gespreksgeschiedenis om context te geven
        history
      }),
    });

    if (!res.ok) {
      const errTxt = await res.text();
      throw new Error(`HTTP ${res.status}: ${errTxt}`);
    }

    const data = await res.json();
    const answer = data.content || data.answer || data.output || "[Leeg antwoord]";

    // UI + history (assistant)
    renderTyping(false);
    renderMessage("assistant", answer);
    history.push({ role: "assistant", content: answer });
    saveHistory();

  } catch (err) {
    renderTyping(false);
    renderMessage("assistant", `⚠️ Fout: ${err.message}`);
  } finally {
    $send.disabled = false;
  }
});

// Handig: Cmd/Ctrl+K wist de chat lokaal
window.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    history = [];
    saveHistory();
    renderFromHistory();
  }
});

