import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";

import { api } from "./api";
import { applyBackendEvents } from "./message-helpers";
import type { BootstrapPayload, SessionSummary, SettingsPayload, ThreadMessage } from "./types";
import { formatPreview, toNumberOrUndefined, uid } from "./utils";

type StatusTone = "idle" | "busy" | "error";

interface ToastState {
  text: string;
  tone: StatusTone;
}

interface AgentFormState {
  label: string;
  provider: string;
  modelName: string;
  temperature: string;
  reasoning: string;
  maxTokens: string;
  contextWindowTokens: string;
}

interface ProviderFormState {
  name: string;
  backend: string;
  apiBase: string;
  apiKey: string;
}

const EMPTY_AGENT_FORM: AgentFormState = {
  label: "",
  provider: "",
  modelName: "",
  temperature: "0.7",
  reasoning: "none",
  maxTokens: "",
  contextWindowTokens: "",
};

const EMPTY_PROVIDER_FORM: ProviderFormState = {
  name: "",
  backend: "",
  apiBase: "",
  apiKey: "",
};

function App() {
  const [bootstrap, setBootstrap] = useState<BootstrapPayload | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null);
  const [messages, setMessages] = useState<ThreadMessage[]>([]);
  const [settings, setSettings] = useState<SettingsPayload | null>(null);
  const [selectedAgentName, setSelectedAgentName] = useState("");
  const [selectedProviderName, setSelectedProviderName] = useState("");
  const [agentForm, setAgentForm] = useState<AgentFormState>(EMPTY_AGENT_FORM);
  const [providerForm, setProviderForm] = useState<ProviderFormState>(EMPTY_PROVIDER_FORM);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState("启动中");
  const [toast, setToast] = useState<ToastState>({ text: "", tone: "idle" });
  const [loadError, setLoadError] = useState<string | null>(null);
  const messageListRef = useRef<HTMLElement | null>(null);
  const toastTimerRef = useRef<number | null>(null);

  const activeSession = sessions.find((item) => item.key === activeSessionKey) ?? null;
  const selectedAgent = settings?.agents.find((item) => item.name === selectedAgentName) ?? settings?.agent ?? null;
  const selectedProvider =
    settings?.providers.find((item) => item.name === selectedProviderName) ??
    settings?.providers.find((item) => item.name === selectedAgent?.provider) ??
    null;

  useEffect(() => {
    void boot();

    return () => {
      if (toastTimerRef.current) {
        window.clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    messageListRef.current?.scrollTo({
      top: messageListRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  useEffect(() => {
    if (!selectedAgent) {
      setAgentForm(EMPTY_AGENT_FORM);
      return;
    }

    setAgentForm({
      label: selectedAgent.label ?? "",
      provider: selectedAgent.provider ?? "",
      modelName: selectedAgent.model_name ?? selectedAgent.model ?? "",
      temperature: selectedAgent.temperature?.toString() ?? "0.7",
      reasoning: selectedAgent.reasoning_effort ?? "none",
      maxTokens: selectedAgent.max_tokens?.toString() ?? "",
      contextWindowTokens: selectedAgent.context_window_tokens?.toString() ?? "",
    });
    setSelectedProviderName(selectedAgent.provider ?? "");
  }, [selectedAgent]);

  useEffect(() => {
    if (!selectedProvider) {
      setProviderForm(EMPTY_PROVIDER_FORM);
      return;
    }

    setProviderForm({
      name: selectedProvider.name,
      backend: selectedProvider.backend ?? selectedProvider.provider_type ?? "",
      apiBase: selectedProvider.api_base ?? selectedProvider.default_api_base ?? "",
      apiKey: "",
    });
  }, [selectedProvider]);

  async function boot() {
    try {
      setBusy(true);
      setStatusText("连接中");

      const [bootstrapPayload, settingsPayload] = await Promise.all([api.bootstrap(), api.loadSettings()]);
      setBootstrap(bootstrapPayload);
      setSettings(settingsPayload);
      setSelectedAgentName(settingsPayload.active_agent ?? settingsPayload.agents[0]?.name ?? "");
      setSelectedProviderName(settingsPayload.agent?.provider ?? settingsPayload.providers[0]?.name ?? "");

      const sessionPayload = await api.listSessions();
      setSessions(sessionPayload.sessions);

      const firstKey = sessionPayload.sessions[0]?.key ?? null;
      if (firstKey) {
        setActiveSessionKey(firstKey);
        const thread = await api.fetchThread(firstKey);
        setMessages(normalizeMessages(thread.messages));
      } else {
        const created = await createSession(false);
        setActiveSessionKey(created.key);
      }

      setStatusText("已连接");
      setLoadError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "前端初始化失败";
      setLoadError(message);
      setStatusText("错误");
      pushToast(message, "error");
    } finally {
      setBusy(false);
    }
  }

  function pushToast(text: string, tone: StatusTone = "idle") {
    setToast({ text, tone });
    if (toastTimerRef.current) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => {
      setToast({ text: "", tone: "idle" });
    }, 3200);
  }

  function normalizeMessages(source: ThreadMessage[]): ThreadMessage[] {
    return source.map((message) => ({
      ...message,
      reasoning: message.reasoning ?? "",
      is_streaming: message.is_streaming ?? false,
    }));
  }

  async function refreshSessions(preferExisting = true) {
    const payload = await api.listSessions();
    setSessions(payload.sessions);

    if (!payload.sessions.length) {
      setActiveSessionKey(null);
      setMessages([]);
      return;
    }

    const nextKey = preferExisting && activeSessionKey ? activeSessionKey : payload.sessions[0].key;
    if (nextKey && nextKey !== activeSessionKey) {
      setActiveSessionKey(nextKey);
    }
  }

  async function openSession(sessionKey: string) {
    setActiveSessionKey(sessionKey);
    const thread = await api.fetchThread(sessionKey);
    setMessages(normalizeMessages(thread.messages));
  }

  async function createSession(showToastAfter = true): Promise<SessionSummary> {
    const title = `Paper desk ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
    const payload = await api.createSession(title);
    await refreshSessions(false);
    setActiveSessionKey(payload.session.key);
    setMessages([]);
    if (showToastAfter) {
      pushToast("已创建新会话");
    }
    return payload.session;
  }

  async function handleCreateSession() {
    try {
      setBusy(true);
      await createSession(true);
    } catch (error) {
      pushToast(error instanceof Error ? error.message : "创建会话失败", "error");
    } finally {
      setBusy(false);
    }
  }

  async function handleSendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = prompt.trim();
    if (!content || !activeSessionKey || busy) {
      return;
    }

    const turnId = uid();
    const optimisticUserMessage: ThreadMessage = {
      id: uid(),
      role: "user",
      content,
      turn_id: turnId,
    };

    setPrompt("");
    setBusy(true);
    setStatusText("运行中");
    setMessages((current) => [...current, optimisticUserMessage]);

    try {
      const payload = await api.sendMessage(activeSessionKey, content, turnId);
      setMessages((current) => applyBackendEvents(current, payload.events));
      await refreshSessions(true);
      pushToast("后端已返回当前轮次结果");
    } catch (error) {
      pushToast(error instanceof Error ? error.message : "发送失败", "error");
    } finally {
      setBusy(false);
      setStatusText("已连接");
    }
  }

  function handlePromptKeydown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function handleRefreshSettings() {
    try {
      const payload = await api.loadSettings();
      setSettings(payload);
      setSelectedAgentName(payload.active_agent ?? payload.agents[0]?.name ?? "");
      pushToast("设置已刷新");
    } catch (error) {
      pushToast(error instanceof Error ? error.message : "刷新设置失败", "error");
    }
  }

  async function handleSaveAgent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedAgentName) {
      return;
    }

    try {
      const payload = await api.saveAgent(selectedAgentName, {
        label: agentForm.label.trim(),
        provider: agentForm.provider,
        model_name: agentForm.modelName.trim(),
        temperature: toNumberOrUndefined(agentForm.temperature),
        reasoning_effort: agentForm.reasoning,
        max_tokens: toNumberOrUndefined(agentForm.maxTokens),
        context_window_tokens: toNumberOrUndefined(agentForm.contextWindowTokens),
      });
      setSettings(payload);
      pushToast("Agent 设置已保存");
    } catch (error) {
      pushToast(error instanceof Error ? error.message : "保存 Agent 失败", "error");
    }
  }

  async function handleSaveProvider(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!providerForm.name) {
      return;
    }

    const patch: Record<string, unknown> = {
      backend: providerForm.backend,
      api_base: providerForm.apiBase.trim(),
    };
    if (providerForm.apiKey.trim()) {
      patch.api_key = providerForm.apiKey.trim();
    }

    try {
      const payload = await api.saveProvider(providerForm.name, patch);
      setSettings(payload);
      pushToast("Provider 连接已保存");
    } catch (error) {
      pushToast(error instanceof Error ? error.message : "保存 Provider 失败", "error");
    }
  }

  const runtimeCaps = bootstrap?.runtime_capabilities ?? {};

  return (
    <div className="shell">
      <aside className="rail" aria-label="会话列表">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            PA
          </div>
          <div>
            <p className="eyebrow">Research relay</p>
            <h1>论文工作台</h1>
          </div>
        </div>

        <button className="command command-primary" type="button" onClick={handleCreateSession} disabled={busy}>
          <span aria-hidden="true">+</span>
          新建会话
        </button>

        <div className="rail-section">
          <div className="section-label">最近会话</div>
          <div className="session-list" role="list">
            {sessions.length ? (
              sessions.map((session) => (
                <button
                  className={`session-item ${session.key === activeSessionKey ? "active" : ""}`}
                  key={session.key}
                  type="button"
                  onClick={() => void openSession(session.key)}
                >
                  <strong>{session.title || "New chat"}</strong>
                  <span>{formatPreview(session.preview, "等待第一条消息")}</span>
                </button>
              ))
            ) : (
              <button className="session-item active" type="button">
                <strong>还没有会话</strong>
                <span>先创建一个论文阅读线程</span>
              </button>
            )}
          </div>
        </div>

        <div className="runtime-panel">
          <div className="section-label">运行面板</div>
          <div className="runtime-grid">
            <div className="runtime-chip">
              <span>REST</span>
              <b>{runtimeCaps.fastapi_rest === true ? "on" : "off"}</b>
            </div>
            <div className="runtime-chip">
              <span>HTTP message</span>
              <b>{runtimeCaps.http_message_submit === true ? "on" : "off"}</b>
            </div>
            <div className="runtime-chip">
              <span>Settings</span>
              <b>{runtimeCaps.settings_snapshot === true ? "on" : "off"}</b>
            </div>
            <div className="runtime-chip">
              <span>Auth</span>
              <b>{runtimeCaps.auth_required ? "required" : "local"}</b>
            </div>
          </div>
        </div>
      </aside>

      <main className="conversation" aria-label="论文智能体对话区">
        <header className="desk-header">
          <div>
            <p className="eyebrow">FastAPI gateway</p>
            <h2>{loadError ? "后端连接失败" : activeSession?.title || "正在连接后端"}</h2>
          </div>
          <div className={`status-pill ${loadError ? "error" : busy ? "busy" : ""}`}>{statusText}</div>
        </header>

        <section className="message-list" aria-live="polite" ref={messageListRef}>
          {!activeSessionKey ? (
            <div className="empty-state">
              <div className="empty-state-inner">
                <h3>把论文问题摆上工作台</h3>
                <p>创建会话后，可以直接向后端提交问题，这里会展示用户消息、推理片段和最终回复。</p>
              </div>
            </div>
          ) : !messages.length ? (
            <div className="empty-state">
              <div className="empty-state-inner">
                <h3>这条线程还没有内容</h3>
                <p>试着让 Agent 总结摘要、方法、实验结论，或者列出需要复核的证据链。</p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-role">{message.role}</div>
                <div className="message-body">
                  {message.reasoning ? <div className="reasoning">{message.reasoning}</div> : null}
                  {message.content || (message.is_streaming ? "正在整理回答..." : "")}
                </div>
              </article>
            ))
          )}
        </section>

        <form className="composer" onSubmit={handleSendMessage}>
          <label className="sr-only" htmlFor="prompt">
            向论文智能体发送问题
          </label>
          <textarea
            id="prompt"
            rows={3}
            placeholder="例如：总结这篇论文的贡献、实验设置和潜在缺陷"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={handlePromptKeydown}
          />
          <div className="composer-actions">
            <span>Enter 换行，Ctrl/Cmd + Enter 发送</span>
            <button className="command command-ink" type="submit" disabled={busy || !activeSessionKey}>
              发送
            </button>
          </div>
        </form>
      </main>

      <aside className="settings" aria-label="模型设置">
        <div className="settings-head">
          <div>
            <p className="eyebrow">Model control</p>
            <h2>当前 Agent</h2>
          </div>
          <button className="icon-button" type="button" aria-label="刷新设置" onClick={() => void handleRefreshSettings()}>
            ↻
          </button>
        </div>

        <form className="settings-form" onSubmit={handleSaveAgent}>
          <label>
            Agent
            <select value={selectedAgentName} onChange={(event) => setSelectedAgentName(event.target.value)}>
              {settings?.agents.map((agent) => (
                <option key={agent.name} value={agent.name}>
                  {agent.label || agent.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            显示名称
            <input
              name="label"
              type="text"
              value={agentForm.label}
              onChange={(event) => setAgentForm((current) => ({ ...current, label: event.target.value }))}
            />
          </label>
          <label>
            Provider
            <select
              value={agentForm.provider}
              onChange={(event) => setAgentForm((current) => ({ ...current, provider: event.target.value }))}
            >
              {settings?.providers.map((provider) => (
                <option key={provider.name} value={provider.name}>
                  {provider.label || provider.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            模型
            <input
              name="model"
              type="text"
              value={agentForm.modelName}
              onChange={(event) => setAgentForm((current) => ({ ...current, modelName: event.target.value }))}
            />
          </label>
          <div className="split-fields">
            <label>
              温度
              <input
                name="temperature"
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={agentForm.temperature}
                onChange={(event) => setAgentForm((current) => ({ ...current, temperature: event.target.value }))}
              />
            </label>
            <label>
              推理
              <select
                value={agentForm.reasoning}
                onChange={(event) => setAgentForm((current) => ({ ...current, reasoning: event.target.value }))}
              >
                <option value="none">none</option>
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="high">high</option>
              </select>
            </label>
          </div>
          <div className="split-fields">
            <label>
              Max tokens
              <input
                name="max_tokens"
                type="number"
                min="1"
                step="1"
                value={agentForm.maxTokens}
                onChange={(event) => setAgentForm((current) => ({ ...current, maxTokens: event.target.value }))}
              />
            </label>
            <label>
              Context
              <input
                name="context_window_tokens"
                type="number"
                min="1"
                step="1"
                value={agentForm.contextWindowTokens}
                onChange={(event) => setAgentForm((current) => ({ ...current, contextWindowTokens: event.target.value }))}
              />
            </label>
          </div>
          <button className="command command-primary" type="submit">
            保存 Agent
          </button>
        </form>

        <form className="settings-form provider-form" onSubmit={handleSaveProvider}>
          <div className="section-label">Provider 连接</div>
          <label>
            Provider
            <select value={selectedProviderName} onChange={(event) => setSelectedProviderName(event.target.value)}>
              {settings?.providers.map((provider) => (
                <option key={provider.name} value={provider.name}>
                  {provider.label || provider.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            类型
            <select
              value={providerForm.backend}
              onChange={(event) => setProviderForm((current) => ({ ...current, backend: event.target.value }))}
            >
              {settings?.provider_types.map((providerType) => (
                <option key={providerType.backend || providerType.name} value={providerType.backend || providerType.name}>
                  {providerType.label || providerType.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            API Base
            <input
              name="api_base"
              type="url"
              placeholder="https://..."
              value={providerForm.apiBase}
              onChange={(event) => setProviderForm((current) => ({ ...current, apiBase: event.target.value }))}
            />
          </label>
          <label>
            API Key
            <input
              name="api_key"
              type="password"
              placeholder="留空则不覆盖现有 Key"
              value={providerForm.apiKey}
              onChange={(event) => setProviderForm((current) => ({ ...current, apiKey: event.target.value }))}
            />
          </label>
          <button className="command command-ink" type="submit">
            保存 Provider
          </button>
        </form>

        <div className={`toast ${toast.tone}`} role="status" aria-live="polite">
          {toast.text}
        </div>
      </aside>
    </div>
  );
}

export default App;
