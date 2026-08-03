export interface AgentItem {
  name: string;
  label: string;
  is_default: boolean;
  model: string;
  model_name: string;
  provider: string;
  resolved_provider: string;
  description: string;
  temperature: number | null;
  reasoning_effort: string | null;
  reasoning_effort_values: string[];
}

export interface EmbeddingProfileItem {
  name: string;
  label: string;
  is_default: boolean;
  provider: string;
  model: string;
  model_name: string;
  dimensions: number | null;
  batch_size: number | null;
}

export interface ProviderEditableConfig {
  backend: string;
  api_key: string | null;
  api_key_env: string | null;
  api_base: string | null;
  extra_headers: Record<string, string>;
  extra_body: Record<string, unknown>;
}

export interface ProviderItem {
  name: string;
  label: string;
  configured: boolean;
  auth_type: string;
  api_key_required: boolean;
  api_key_hint: string | null;
  api_key_env: string | null;
  api_base: string | null;
  default_api_base: string;
  model_selectable: boolean;
  provider_type: string;
  backend: string;
  oauth_login_supported: boolean;
  editable_config: ProviderEditableConfig;
}

export interface ProviderTypeItem {
  name: string;
  label: string;
  backend: string;
  default_api_base: string;
  api_key_required: boolean;
}

export interface DefaultsPayload {
  llm: {
    temperature: number | null;
    max_tokens: number | null;
    reasoning_effort: string | null;
    context_window_tokens: number | null;
  };
  embedding: {
    dimensions: number | null;
    batch_size: number | null;
  };
}

export interface SettingsPayload {
  active_agent: string;
  agents: AgentItem[];
  providers: ProviderItem[];
  provider_types: ProviderTypeItem[];
  embedding_profiles: EmbeddingProfileItem[];
  defaults: DefaultsPayload;
  requires_restart: boolean;
  restart_required_sections: string[];
  apply_state: string;
  runtime_capabilities: Record<string, boolean>;
  surface: string;
}

export interface ProviderModelItem {
  id: string;
  label: string;
  owned_by: string | null;
  context_window: number | null;
}

export interface ProviderModelsPayload {
  provider: string;
  label: string;
  status: "available" | "missing_api_base" | "not_configured" | "unsupported" | "error";
  catalog_kind: string;
  models: ProviderModelItem[];
  model_count: number;
  message: string;
  fetched_at: string;
}

export interface ModelConnectivityPayload {
  target_type: "agent" | "embedding_profile";
  name: string;
  provider: string;
  model: string;
  status: "passed" | "failed" | "not_configured";
  message: string;
  latency_ms: number;
  error_kind: string | null;
  error_status_code: number | null;
  finish_reason: string | null;
  vector_dimensions: number | null;
  tested_at: string;
}
