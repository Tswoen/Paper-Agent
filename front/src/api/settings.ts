import type {
  ProviderModelsPayload,
  SettingsPayload,
} from "../types/settings";

type JsonObject = Record<string, unknown>;

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

export function getSettings(): Promise<SettingsPayload> {
  return request<SettingsPayload>("/api/settings");
}

export function saveProvider(
  name: string,
  payload: JsonObject,
): Promise<SettingsPayload> {
  return request<SettingsPayload>(`/api/settings/providers/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function saveAgent(
  name: string,
  payload: JsonObject,
): Promise<SettingsPayload> {
  return request<SettingsPayload>(`/api/settings/agents/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function saveEmbeddingProfile(
  name: string,
  payload: JsonObject,
): Promise<SettingsPayload> {
  return request<SettingsPayload>(
    `/api/settings/embedding-profiles/${encodeURIComponent(name)}`,
    {
      method: "PUT",
      body: JSON.stringify(payload),
    },
  );
}

export function getProviderModels(
  provider: string,
): Promise<ProviderModelsPayload> {
  return request<ProviderModelsPayload>(
    `/api/settings/provider-models?provider=${encodeURIComponent(provider)}`,
  );
}
