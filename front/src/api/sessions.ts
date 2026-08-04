import type {
  SessionCreatePayload,
  SessionListPayload,
  SessionRunAccepted,
  SessionRunStartPayload,
  SessionRuntimeEvent,
  SessionThreadPayload,
} from "../types/sessions";

type JsonObject = Record<string, unknown>;

export class ApiRequestError extends Error {
  status: number;

  /** 中文注释：把后端状态码一起保存下来，页面就能判断是 404、409 还是普通网络错误。 */
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

const STREAM_EVENTS = [
  "runtime_event",
  "message",
  "reasoning_delta",
  "reasoning_end",
  "delta",
  "tool",
  "artifact",
  "status",
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
    detail?: string | { message?: string };
  } & T;

  if (!response.ok) {
    const detailMessage = typeof data.detail === "string" ? data.detail : data.detail?.message;
    // 中文注释：这里保留 HTTP 状态码，调用方遇到“会话不存在”时可以主动把界面切回空白页。
    throw new ApiRequestError(data.error?.message || detailMessage || "请求失败", response.status);
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
  payload: SessionRunStartPayload,
): Promise<SessionRunAccepted> {
  return request<SessionRunAccepted>(`/api/sessions/${encodeURIComponent(sessionKey)}/runs`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function cancelSessionRun(
  sessionKey: string,
  runId: string,
): Promise<{ session_key: string; run_id: string; status: string }> {
  return request<{ session_key: string; run_id: string; status: string }>(
    `/api/sessions/${encodeURIComponent(sessionKey)}/runs/${encodeURIComponent(runId)}/cancel`,
    { method: "POST", body: JSON.stringify({}) },
  );
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
      try {
        handlers.onEvent(JSON.parse(messageEvent.data) as SessionRuntimeEvent);
      } catch {
        // 中文注释：如果某一条 SSE 数据意外不是合法 JSON，只忽略这一条，避免整个实时连接被前端代码打断。
      }
    });
  }

  source.onerror = (event) => {
    handlers.onError?.(event);
  };

  return source;
}
