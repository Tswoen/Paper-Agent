import type {
  SessionCreatePayload,
  SessionListPayload,
  SessionRunAccepted,
  SessionRuntimeEvent,
  SessionThreadPayload,
} from "../types/sessions";

type JsonObject = Record<string, unknown>;

const STREAM_EVENTS = [
  "message",
  "reasoning_delta",
  "reasoning_end",
  "delta",
  "tool",
  "artifact",
  "status",
  "node_started",
  "node_progress",
  "node_completed",
  "node_failed",
  "error",
  "turn_end",
] as const;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  const data = (await response.json().catch(() => ({}))) as {
    error?: { message?: string };
  } & T;

  if (!response.ok) {
    throw new Error(data.error?.message || "请求失败");
  }

  return data as T;
}

export function listSessions(): Promise<SessionListPayload> {
  return request<SessionListPayload>("/api/sessions");
}

export function createSession(payload?: JsonObject): Promise<SessionCreatePayload> {
  return request<SessionCreatePayload>("/api/sessions", {
    method: "POST",
    body: JSON.stringify(payload ?? {}),
  });
}

export function fetchSessionThread(sessionKey: string): Promise<SessionThreadPayload> {
  return request<SessionThreadPayload>(`/api/sessions/${encodeURIComponent(sessionKey)}/webui-thread`);
}

export function deleteSession(sessionKey: string): Promise<{ deleted: boolean; key: string }> {
  return request<{ deleted: boolean; key: string }>(`/api/sessions/${encodeURIComponent(sessionKey)}`, {
    method: "DELETE",
  });
}

export function startSessionRun(
  sessionKey: string,
  payload: { content: string; turn_id?: string },
): Promise<SessionRunAccepted> {
  return request<SessionRunAccepted>(`/api/sessions/${encodeURIComponent(sessionKey)}/runs`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function subscribeSessionRun(
  streamUrl: string,
  handlers: {
    onEvent: (event: SessionRuntimeEvent) => void;
    onError?: (error: Event) => void;
    onOpen?: () => void;
  },
): EventSource {
  const source = new EventSource(streamUrl);

  if (handlers.onOpen) {
    source.onopen = () => {
      handlers.onOpen?.();
    };
  }

  for (const eventName of STREAM_EVENTS) {
    source.addEventListener(eventName, (event) => {
      const messageEvent = event as MessageEvent<string>;
      handlers.onEvent(JSON.parse(messageEvent.data) as SessionRuntimeEvent);
    });
  }

  source.onerror = (event) => {
    handlers.onError?.(event);
  };

  return source;
}
