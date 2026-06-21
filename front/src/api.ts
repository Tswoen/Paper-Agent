import type {
  BootstrapPayload,
  MessageSubmitPayload,
  SessionCreatePayload,
  SessionListPayload,
  SettingsPayload,
  ThreadPayload,
} from "./types";

interface ApiErrorPayload {
  error?: {
    message?: string;
  };
  detail?: {
    message?: string;
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  const payload = (await response.json().catch(() => ({}))) as ApiErrorPayload & T;
  if (!response.ok) {
    throw new Error(payload.error?.message || payload.detail?.message || `请求失败：${response.status}`);
  }
  return payload;
}

export const api = {
  bootstrap(): Promise<BootstrapPayload> {
    return request<BootstrapPayload>("/webui/bootstrap");
  },
  listSessions(): Promise<SessionListPayload> {
    return request<SessionListPayload>("/api/sessions");
  },
  createSession(title: string): Promise<SessionCreatePayload> {
    return request<SessionCreatePayload>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        title,
        workspace_scope: { source: "front/workbench-react" },
      }),
    });
  },
  fetchThread(sessionKey: string): Promise<ThreadPayload> {
    return request<ThreadPayload>(`/api/sessions/${encodeURIComponent(sessionKey)}/webui-thread`);
  },
  sendMessage(sessionKey: string, content: string, turnId: string): Promise<MessageSubmitPayload> {
    return request<MessageSubmitPayload>(`/api/sessions/${encodeURIComponent(sessionKey)}/messages`, {
      method: "POST",
      body: JSON.stringify({
        content,
        turn_id: turnId,
      }),
    });
  },
  loadSettings(): Promise<SettingsPayload> {
    return request<SettingsPayload>("/api/settings");
  },
  saveAgent(name: string, patch: Record<string, unknown>): Promise<SettingsPayload> {
    return request<SettingsPayload>(`/api/settings/agents/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(patch),
    });
  },
  saveProvider(name: string, patch: Record<string, unknown>): Promise<SettingsPayload> {
    return request<SettingsPayload>(`/api/settings/providers/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(patch),
    });
  },
};
