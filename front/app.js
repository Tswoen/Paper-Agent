const state = {
  apiBase: "",
  bootstrap: null,
  sessions: [],
  activeSessionKey: null,
  messages: [],
  settings: null,
  isBusy: false,
};

const els = {
  activeTitle: document.querySelector("#active-title"),
  agentContext: document.querySelector("#agent-context"),
  agentForm: document.querySelector("#agent-form"),
  agentLabel: document.querySelector("#agent-label"),
  agentMaxTokens: document.querySelector("#agent-max-tokens"),
  agentModel: document.querySelector("#agent-model"),
  agentProvider: document.querySelector("#agent-provider"),
  agentReasoning: document.querySelector("#agent-reasoning"),
  agentSelect: document.querySelector("#agent-select"),
  agentTemperature: document.querySelector("#agent-temperature"),
  composer: document.querySelector("#composer"),
  messageList: document.querySelector("#message-list"),
  newSession: document.querySelector("#new-session"),
  prompt: document.querySelector("#prompt"),
  providerApiBase: document.querySelector("#provider-api-base"),
  providerApiKey: document.querySelector("#provider-api-key"),
  providerForm: document.querySelector("#provider-form"),
  providerSelect: document.querySelector("#provider-select"),
  providerType: document.querySelector("#provider-type"),
  refreshSettings: document.querySelector("#refresh-settings"),
  runtimeGrid: document.querySelector("#runtime-grid"),
  sendMessage: document.querySelector("#send-message"),
  sessionList: document.querySelector("#session-list"),
  statusPill: document.querySelector("#status-pill"),
  toast: document.querySelector("#toast"),
};

async function request(path, options = {}) {
  const response = await fetch(`${state.apiBase}${path}`, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.error?.message || payload?.detail?.message || `请求失败：${response.status}`;
    throw new Error(message);
  }
  return payload;
}

function uid() {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function showToast(message) {
  els.toast.textContent = message;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    els.toast.textContent = "";
  }, 3200);
}

function setBusy(isBusy, label = "就绪") {
  state.isBusy = isBusy;
  els.sendMessage.disabled = isBusy;
  els.statusPill.textContent = isBusy ? "运行中" : label;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderRuntime() {
  const caps = state.bootstrap?.runtime_capabilities || {};
  const rows = [
    ["REST", caps.fastapi_rest],
    ["HTTP message", caps.http_message_submit],
    ["Settings", caps.settings_snapshot],
    ["Auth", caps.auth_required ? "required" : "local"],
  ];
  els.runtimeGrid.innerHTML = rows
    .map(([label, value]) => `<div class="runtime-chip"><span>${label}</span><b>${value === true ? "on" : value || "off"}</b></div>`)
    .join("");
}

function renderSessions() {
  if (!state.sessions.length) {
    els.sessionList.innerHTML = `<button class="session-item active" type="button"><strong>还没有会话</strong><span>创建一个论文阅读线程</span></button>`;
    return;
  }
  els.sessionList.innerHTML = state.sessions
    .map((session) => {
      const active = session.key === state.activeSessionKey ? " active" : "";
      const preview = session.preview || "等待第一条问题";
      return `
        <button class="session-item${active}" type="button" data-session-key="${escapeHtml(session.key)}">
          <strong>${escapeHtml(session.title || "New chat")}</strong>
          <span>${escapeHtml(preview)}</span>
        </button>`;
    })
    .join("");
}

function renderMessages() {
  const active = state.sessions.find((item) => item.key === state.activeSessionKey);
  els.activeTitle.textContent = active?.title || "论文工作台";

  if (!state.activeSessionKey) {
    els.messageList.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-inner">
          <h3>把论文问题放到桌面上</h3>
          <p>创建会话后，可以直接向后端提交问题。这里会展示用户消息、推理片段和最终回复。</p>
        </div>
      </div>`;
    return;
  }

  if (!state.messages.length) {
    els.messageList.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-inner">
          <h3>这条线程还是空白</h3>
          <p>试着让 Agent 提炼摘要、方法、实验结论，或要求它列出需要复核的证据。</p>
        </div>
      </div>`;
    return;
  }

  els.messageList.innerHTML = state.messages
    .map((message) => {
      const role = message.role === "assistant" ? "assistant" : message.role === "system" ? "system" : "user";
      const reasoning = message.reasoning ? `<div class="reasoning">${escapeHtml(message.reasoning)}</div>` : "";
      const content = message.content || (message.is_streaming ? "正在整理回答..." : "");
      return `
        <article class="message ${role}">
          <div class="message-role">${role}</div>
          <div class="message-body">${reasoning}${escapeHtml(content)}</div>
        </article>`;
    })
    .join("");
  els.messageList.scrollTop = els.messageList.scrollHeight;
}

function applyBackendEvent(event) {
  const name = event.event;
  if (name === "message") {
    if (event.role === "user") {
      const exists = state.messages.some((item) => item.turn_id && item.turn_id === event.turn_id && item.role === "user");
      if (!exists) {
        state.messages.push({
          id: event.id || uid(),
          role: "user",
          content: event.content || "",
          media: event.media || [],
          turn_id: event.turn_id,
        });
      }
      return;
    }
    const assistant = ensureAssistant(event);
    assistant.content = event.content || assistant.content;
    return;
  }
  if (name === "reasoning_delta") {
    // 后端会把推理增量作为事件返回，这里合并到当前 assistant 气泡中。
    const assistant = ensureAssistant(event);
    assistant.reasoning = `${assistant.reasoning || ""}${event.content || event.delta || ""}`;
    assistant.reasoning_streaming = true;
    return;
  }
  if (name === "delta") {
    // delta 只追加到当前轮 assistant 消息，避免每个片段生成一条新消息。
    const assistant = ensureAssistant(event);
    assistant.content = `${assistant.content || ""}${event.content || event.delta || ""}`;
    assistant.is_streaming = true;
    return;
  }
  if (name === "reasoning_end") {
    const assistant = findActiveAssistant(event.turn_id);
    if (assistant) assistant.reasoning_streaming = false;
    return;
  }
  if (name === "stream_end" || name === "turn_end") {
    const assistant = findActiveAssistant(event.turn_id);
    if (assistant) assistant.is_streaming = false;
  }
}

function findActiveAssistant(turnId) {
  const candidates = state.messages.filter((item) => item.role === "assistant");
  for (let index = candidates.length - 1; index >= 0; index -= 1) {
    if (candidates[index].turn_id === turnId) {
      return candidates[index];
    }
  }
  return null;
}

function ensureAssistant(event) {
  let assistant = findActiveAssistant(event.turn_id);
  if (!assistant) {
    assistant = {
      id: uid(),
      role: "assistant",
      content: "",
      reasoning: "",
      is_streaming: true,
      turn_id: event.turn_id,
    };
    state.messages.push(assistant);
  }
  return assistant;
}

async function loadBootstrap() {
  state.bootstrap = await request("/webui/bootstrap");
  state.apiBase = state.bootstrap.api_base || "";
  renderRuntime();
}

async function loadSessions(selectFirst = true) {
  const payload = await request("/api/sessions");
  state.sessions = payload.sessions || [];
  if (selectFirst && !state.activeSessionKey && state.sessions[0]) {
    state.activeSessionKey = state.sessions[0].key;
    await loadThread(state.activeSessionKey);
  }
  renderSessions();
  renderMessages();
}

async function loadThread(sessionKey) {
  const payload = await request(`/api/sessions/${encodeURIComponent(sessionKey)}/webui-thread`);
  state.activeSessionKey = sessionKey;
  state.messages = (payload.messages || []).map((message) => ({
    ...message,
    reasoning: message.reasoning || "",
  }));
  renderSessions();
  renderMessages();
}

async function createSession() {
  const title = `Paper desk ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
  const payload = await request("/api/sessions", {
    method: "POST",
    body: JSON.stringify({
      title,
      workspace_scope: { source: "front/workbench" },
    }),
  });
  state.activeSessionKey = payload.session.key;
  state.messages = [];
  await loadSessions(false);
  renderMessages();
  showToast("已创建新会话");
}

async function sendMessage(event) {
  event.preventDefault();
  const content = els.prompt.value.trim();
  if (!content || !state.activeSessionKey || state.isBusy) return;

  const turnId = uid();
  state.messages.push({ id: uid(), role: "user", content, turn_id: turnId });
  els.prompt.value = "";
  setBusy(true);
  renderMessages();

  try {
    const payload = await request(`/api/sessions/${encodeURIComponent(state.activeSessionKey)}/messages`, {
      method: "POST",
      body: JSON.stringify({ content, turn_id: turnId }),
    });
    for (const backendEvent of payload.events || []) {
      applyBackendEvent(backendEvent);
    }
    await loadSessions(false);
    renderMessages();
    showToast("后端已返回本轮事件");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function fillSelect(select, items, getValue, getLabel) {
  select.innerHTML = items
    .map((item) => `<option value="${escapeHtml(getValue(item))}">${escapeHtml(getLabel(item))}</option>`)
    .join("");
}

async function loadSettings() {
  state.settings = await request("/api/settings");
  const agents = state.settings.agents || [];
  const providers = state.settings.providers || [];
  const providerTypes = state.settings.provider_types || [];

  fillSelect(els.agentSelect, agents, (item) => item.name, (item) => item.label || item.name);
  fillSelect(els.agentProvider, providers, (item) => item.name, (item) => item.label || item.name);
  fillSelect(els.providerSelect, providers, (item) => item.name, (item) => item.label || item.name);
  fillSelect(els.providerType, providerTypes, (item) => item.backend || item.name, (item) => item.label || item.name);

  els.agentSelect.value = state.settings.active_agent || agents[0]?.name || "";
  renderAgentForm();
  renderProviderForm();
}

function selectedAgent() {
  const name = els.agentSelect.value || state.settings?.active_agent;
  return (state.settings?.agents || []).find((item) => item.name === name) || state.settings?.agent;
}

function selectedProvider() {
  const name = els.providerSelect.value || selectedAgent()?.provider;
  return (state.settings?.providers || []).find((item) => item.name === name);
}

function renderAgentForm() {
  const agent = selectedAgent();
  if (!agent) return;
  els.agentLabel.value = agent.label || "";
  els.agentProvider.value = agent.provider || "";
  els.agentModel.value = agent.model_name || agent.model || "";
  els.agentTemperature.value = agent.temperature ?? 0.7;
  els.agentReasoning.value = agent.reasoning_effort || "none";
  els.agentMaxTokens.value = agent.max_tokens || "";
  els.agentContext.value = agent.context_window_tokens || "";
  els.providerSelect.value = agent.provider || els.providerSelect.value;
  renderProviderForm();
}

function renderProviderForm() {
  const provider = selectedProvider();
  if (!provider) return;
  els.providerType.value = provider.backend || provider.provider_type || "openai_compat";
  els.providerApiBase.value = provider.api_base || provider.default_api_base || "";
  els.providerApiKey.value = "";
}

async function saveAgent(event) {
  event.preventDefault();
  const name = els.agentSelect.value || "default_agent";
  const patch = {
    label: els.agentLabel.value.trim(),
    provider: els.agentProvider.value,
    model_name: els.agentModel.value.trim(),
    temperature: Number(els.agentTemperature.value),
    reasoning_effort: els.agentReasoning.value,
    max_tokens: Number(els.agentMaxTokens.value),
    context_window_tokens: Number(els.agentContext.value),
  };
  try {
    state.settings = await request(`/api/settings/agents/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(patch),
    });
    await loadSettings();
    showToast("Agent 设置已保存");
  } catch (error) {
    showToast(error.message);
  }
}

async function saveProvider(event) {
  event.preventDefault();
  const name = els.providerSelect.value;
  const patch = {
    backend: els.providerType.value,
    api_base: els.providerApiBase.value.trim(),
  };
  if (els.providerApiKey.value.trim()) {
    patch.api_key = els.providerApiKey.value.trim();
  }
  try {
    state.settings = await request(`/api/settings/providers/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(patch),
    });
    await loadSettings();
    showToast("Provider 连接已保存");
  } catch (error) {
    showToast(error.message);
  }
}

function bindEvents() {
  els.newSession.addEventListener("click", createSession);
  els.sessionList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-session-key]");
    if (button) loadThread(button.dataset.sessionKey);
  });
  els.composer.addEventListener("submit", sendMessage);
  els.prompt.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      els.composer.requestSubmit();
    }
  });
  els.refreshSettings.addEventListener("click", () => loadSettings().then(() => showToast("设置已刷新")));
  els.agentSelect.addEventListener("change", renderAgentForm);
  els.providerSelect.addEventListener("change", renderProviderForm);
  els.agentForm.addEventListener("submit", saveAgent);
  els.providerForm.addEventListener("submit", saveProvider);
}

async function boot() {
  bindEvents();
  try {
    await loadBootstrap();
    await Promise.all([loadSessions(), loadSettings()]);
    if (!state.activeSessionKey) {
      await createSession();
    }
    setBusy(false, "已连接");
  } catch (error) {
    els.activeTitle.textContent = "后端连接失败";
    els.statusPill.textContent = "错误";
    showToast(error.message);
    renderMessages();
  }
}

boot();
