let selectedModel = 'svm';
let isLoading = false;

const STYLE_COLORS = {
  'Научный':             '#7C6EF0',
  'Официально-деловой':  '#3DBF8C',
  'Публицистический':    '#4A9EE0',
  'Разговорный':         '#E06060',
  'Художественный':      '#D4900A',
};

document.querySelector('[data-model="svm"]').classList.add('active');

function selectModel(btn) {
  document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedModel = btn.dataset.model;
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function addMessage(type, content) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const chat = document.getElementById('chat');
  const div  = document.createElement('div');
  div.className = `msg ${type}`;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = type === 'user' ? 'Вы' : 'Модель';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  if (typeof content === 'string') {
    bubble.textContent = content;
  } else {
    bubble.appendChild(content);
    bubble.style.padding = '0';
    bubble.style.background = 'transparent';
    bubble.style.border = 'none';
  }

  div.appendChild(label);
  div.appendChild(bubble);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addTyping() {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const chat = document.getElementById('chat');
  const div  = document.createElement('div');
  div.className = 'msg bot';
  div.id = 'typing';

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = 'Модель';

  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';

  div.appendChild(label);
  div.appendChild(typing);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

function buildResult(data) {
  const card = document.createElement('div');
  card.className = 'result-card';

  card.innerHTML = `
    <div class="result-header">
      <div>
        <div class="result-style" style="color:var(--accent)">${data.predicted_name}</div>
        <div class="result-confidence">Уверенность: ${data.confidence}%</div>
      </div>
    </div>
    <div class="bars-title">Вероятности по стилям</div>
    ${data.all_probs.map(s => `
      <div class="bar-row">
        <div class="bar-label ${s.name === data.predicted_name ? 'active' : ''}">${s.name}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${s.prob}%; background:var(--accent)"></div>
        </div>
        <div class="bar-pct ${s.name === data.predicted_name ? 'active' : ''}">${s.prob}%</div>
      </div>
    `).join('')}
  `;
  return card;
}

async function sendMessage() {
  const ta   = document.getElementById('inputText');
  const text = ta.value.trim();
  if (!text || isLoading) return;

  isLoading = true;
  document.getElementById('sendBtn').disabled = true;

  addMessage('user', text);
  ta.value = '';
  autoResize(ta);
  addTyping();

  try {
    const res  = await fetch('/classify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text, model: selectedModel })
    });
    const data = await res.json();
    removeTyping();

    if (data.error) {
      addMessage('bot', `${data.error}`);
    } else {
      addMessage('bot', buildResult(data));
    }
  } catch (e) {
    removeTyping();
    addMessage('bot', 'Ошибка соединения с сервером');
  }

  isLoading = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('chat').scrollTop = 99999;
}
