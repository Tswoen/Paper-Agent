<script setup lang="ts">
import {
  ArrowUpRight,
  Bot,
  Database,
  LoaderCircle,
  Plus,
  RefreshCw,
  Save,
  SearchCheck,
  ShieldCheck,
  Trash2,
} from "lucide-vue-next";
import { computed, onMounted, reactive, ref, watch } from "vue";

import {
  deleteProvider,
  getProviderModels,
  getSettings,
  saveAgent,
  saveEmbeddingProfile,
  saveProvider,
  testModelConnectivity,
} from "../api/settings";
import StatusPill from "../components/StatusPill.vue";
import { pushToast } from "../stores/notifications";
import type {
  AgentItem,
  EmbeddingProfileItem,
  ModelConnectivityPayload,
  ProviderItem,
  ProviderModelsPayload,
  SettingsPayload,
} from "../types/settings";

type ProviderDraft = {
  backend: string;
  api_key: string;
  api_key_env: string;
  api_base: string;
  extra_headers: string;
  extra_body: string;
};

type AgentDraft = {
  name: string;
  label: string;
  provider: string;
  model_name: string;
  temperature: string;
  reasoning_effort: string;
};

type EmbeddingDraft = {
  name: string;
  label: string;
  provider: string;
  model_name: string;
  dimensions: string;
  batch_size: string;
};

const loading = ref(true);
const refreshing = ref(false);
const savingProvider = ref(false);
const deletingProvider = ref(false);
const savingAgent = ref(false);
const savingEmbedding = ref(false);
const showApiKey = ref(false);
// 控制右侧是否处于新增状态，新增时显示空白表单，已有 Provider 则显示配置详情。
const isCreatingProvider = ref(false);
const isCreatingAgent = ref(false);
const isCreatingEmbedding = ref(false);
const selectedProviderName = ref("");
const editingAgentName = ref("");
const editingEmbeddingName = ref("");
const creatingProvider = reactive({
  name: "",
  backend: "",
});

const settings = ref<SettingsPayload | null>(null);
const providerDraft = reactive<ProviderDraft>({
  backend: "",
  api_key: "",
  api_key_env: "",
  api_base: "",
  extra_headers: "{}",
  extra_body: "{}",
});
const agentDraft = reactive<AgentDraft>({
  name: "",
  label: "",
  provider: "",
  model_name: "",
  temperature: "",
  reasoning_effort: "none",
});
const embeddingDraft = reactive<EmbeddingDraft>({
  name: "",
  label: "",
  provider: "",
  model_name: "",
  dimensions: "",
  batch_size: "",
});

const providerModelsMap = reactive<Record<string, ProviderModelsPayload>>({});
const testingState = reactive<Record<string, boolean>>({});
const connectivityMap = reactive<Record<string, ModelConnectivityPayload>>({});
const connectivityTestingState = reactive<Record<string, boolean>>({});

const providers = computed(() => settings.value?.providers ?? []);
const agents = computed(() => settings.value?.agents ?? []);
const embeddings = computed(() => settings.value?.embedding_profiles ?? []);
const providerTypes = computed(() => settings.value?.provider_types ?? []);

const selectedProvider = computed(() =>
  providers.value.find((item) => item.name === selectedProviderName.value) ?? null,
);
const providerBackend = computed({
  get: () => (isCreatingProvider.value ? creatingProvider.backend : providerDraft.backend),
  set: (value: string) => {
    if (isCreatingProvider.value) {
      creatingProvider.backend = value;
      return;
    }
    providerDraft.backend = value;
  },
});
const activeProviderModels = computed(
  () => providerModelsMap[selectedProviderName.value]?.models ?? [],
);
const selectedProviderType = computed(() =>
  providerTypes.value.find((item) => item.name === providerBackend.value) ?? null,
);
const editingAgent = computed(() =>
  agents.value.find((item) => item.name === editingAgentName.value) ?? null,
);
const editingEmbedding = computed(() =>
  embeddings.value.find((item) => item.name === editingEmbeddingName.value) ?? null,
);
const agentProviderModels = computed(
  () => providerModelsMap[agentDraft.provider]?.models ?? [],
);
const embeddingProviderModels = computed(
  () => providerModelsMap[embeddingDraft.provider]?.models ?? [],
);

const providerCompletion = computed(() => {
  if (!selectedProvider.value) {
    return 0;
  }
  const checks = [
    Boolean(providerDraft.backend),
    Boolean(providerDraft.api_base || selectedProvider.value.default_api_base),
    selectedProvider.value.api_key_required ? Boolean(providerDraft.api_key) : true,
  ];
  return Math.round((checks.filter(Boolean).length / checks.length) * 100);
});

const healthCards = computed(() => {
  const providerCount = providers.value.length;
  const readyProviders = providers.value.filter((item) => item.configured).length;
  const agentCount = agents.value.length;
  const embeddingCount = embeddings.value.length;
  return [
    {
      label: "已配置 Provider",
      value: `${readyProviders}/${providerCount || 0}`,
      detail: "用于模型调用的上游连接",
      icon: ShieldCheck,
    },
    {
      label: "智能体",
      value: String(agentCount),
      detail: "可直接参与编排的模型配置",
      icon: Bot,
    },
    {
      label: "嵌入模型",
      value: String(embeddingCount),
      detail: "向量检索与索引使用的模型",
      icon: Database,
    },
  ];
});

watch(
  providers,
  (list) => {
    if (isCreatingProvider.value) {
      return;
    }
    if (!list.length) {
      selectedProviderName.value = "";
      return;
    }
    // 中文注释：刷新配置后尽量保住用户当前上下文，只有选中项失效时才回退到第一项。
    if (!selectedProviderName.value || !list.some((item) => item.name === selectedProviderName.value)) {
      selectedProviderName.value = list[0].name;
    }
  },
  { immediate: true },
);

watch(selectedProvider, (provider) => {
  if (!provider || isCreatingProvider.value) {
    return;
  }
  syncProviderDraft(provider);
});

watch(editingAgent, (agent) => {
  if (!agent) {
    return;
  }
  syncAgentDraft(agent);
});

watch(editingEmbedding, (profile) => {
  if (!profile) {
    return;
  }
  syncEmbeddingDraft(profile);
});

onMounted(async () => {
  await loadSettings();
});

async function loadSettings() {
  loading.value = true;
  try {
    settings.value = await getSettings();
    resetTransientState();
    primeEditors();
  } catch (error) {
    handleError(error, "加载系统配置失败");
  } finally {
    loading.value = false;
  }
}

async function refreshSettings() {
  refreshing.value = true;
  try {
    settings.value = await getSettings();
    resetTransientState();
    primeEditors();
    pushToast({
      tone: "success",
      title: "配置已刷新",
      description: "页面数据已与后端配置文件重新同步。",
    });
  } catch (error) {
    handleError(error, "刷新配置失败");
  } finally {
    refreshing.value = false;
  }
}

function primeEditors() {
  if (!creatingProvider.backend) {
    creatingProvider.backend = providerTypes.value[0]?.name ?? "";
  }
}

function startCreatingProvider() {
  isCreatingProvider.value = true;
  selectedProviderName.value = "";
  creatingProvider.name = "";
  creatingProvider.backend = providerTypes.value[0]?.name ?? "";
  providerDraft.backend = creatingProvider.backend;
  providerDraft.api_key = "";
  providerDraft.api_key_env = "";
  providerDraft.api_base = selectedProviderType.value?.default_api_base ?? "";
  providerDraft.extra_headers = "{}";
  providerDraft.extra_body = "{}";
}

function selectProvider(name: string) {
  // 点击已有 Provider 时，先退出新增状态，再让 watcher 把对应配置填入右侧表单。
  isCreatingProvider.value = false;
  selectedProviderName.value = name;
}

function cancelCreatingProvider() {
  isCreatingProvider.value = false;
  selectedProviderName.value = providers.value[0]?.name ?? "";
}

function syncProviderDraft(provider: ProviderItem) {
  providerDraft.backend = provider.editable_config.backend || provider.provider_type;
  providerDraft.api_key = provider.editable_config.api_key ?? "";
  providerDraft.api_key_env = provider.editable_config.api_key_env ?? "";
  providerDraft.api_base = provider.editable_config.api_base ?? "";
  providerDraft.extra_headers = prettyJson(provider.editable_config.extra_headers);
  providerDraft.extra_body = prettyJson(provider.editable_config.extra_body);
}

function syncAgentDraft(agent: AgentItem) {
  agentDraft.name = agent.name;
  agentDraft.label = agent.label;
  agentDraft.provider = agent.provider;
  agentDraft.model_name = agent.model_name;
  agentDraft.temperature = toInputString(agent.temperature);
  agentDraft.reasoning_effort = agent.reasoning_effort ?? "none";
}

function syncEmbeddingDraft(profile: EmbeddingProfileItem) {
  embeddingDraft.name = profile.name;
  embeddingDraft.label = profile.label;
  embeddingDraft.provider = profile.provider;
  embeddingDraft.model_name = profile.model_name;
  embeddingDraft.dimensions = toInputString(profile.dimensions);
  embeddingDraft.batch_size = toInputString(profile.batch_size);
}

async function submitProvider() {
  if (!selectedProviderName.value) {
    return;
  }
  savingProvider.value = true;
  try {
    const payload = {
      provider_type: providerDraft.backend,
      api_key: providerDraft.api_key.trim(),
      api_key_env: emptyToUndefined(providerDraft.api_key_env),
      api_base: emptyToUndefined(providerDraft.api_base),
      extra_headers: parseJsonRecord(providerDraft.extra_headers, "额外请求头"),
      extra_body: parseJsonRecord(providerDraft.extra_body, "额外请求体"),
    };
    settings.value = await saveProvider(selectedProviderName.value, payload);
    invalidateProviderDerivedState(selectedProviderName.value);
    primeEditors();
    pushToast({
      tone: "success",
      title: "Provider 已保存",
      description: `${selectedProviderName.value} 的连接配置已更新。`,
    });
  } catch (error) {
    handleError(error, "保存 Provider 失败");
  } finally {
    savingProvider.value = false;
  }
}

async function createProvider() {
  const name = creatingProvider.name.trim();
  const backend = creatingProvider.backend.trim();
  if (!name || !backend) {
    pushToast({
      tone: "error",
      title: "缺少创建参数",
      description: "请填写 Provider ID，并选择一种 Provider 类型。",
    });
    return;
  }
  try {
    settings.value = await saveProvider(name, {
      provider_type: backend,
      api_base:
        emptyToUndefined(providerDraft.api_base) ||
        providerTypes.value.find((item) => item.name === backend)?.default_api_base ||
        undefined,
      api_key: providerDraft.api_key.trim(),
      api_key_env: emptyToUndefined(providerDraft.api_key_env),
      extra_headers: parseJsonRecord(providerDraft.extra_headers, "额外请求头"),
      extra_body: parseJsonRecord(providerDraft.extra_body, "额外请求体"),
    });
    isCreatingProvider.value = false;
    selectedProviderName.value = name;
    creatingProvider.name = "";
    invalidateProviderDerivedState(name);
    primeEditors();
    pushToast({
      tone: "success",
      title: "Provider 已创建",
      description: `${name} 已加入当前配置。`,
    });
  } catch (error) {
    handleError(error, "创建 Provider 失败");
  }
}

async function deleteCurrentProvider() {
  const name = selectedProviderName.value;
  if (!name) {
    return;
  }
  const confirmed = window.confirm(
    `确定删除 Provider「${name}」吗？\n引用它的智能体和嵌入模型也会一起被删除。`,
  );
  if (!confirmed) {
    return;
  }
  deletingProvider.value = true;
  try {
    settings.value = await deleteProvider(name);
    invalidateProviderDerivedState(name);
    // 删完后退出编辑态，让列表选中第一个剩下的 Provider。
    isCreatingProvider.value = false;
    selectedProviderName.value = providers.value[0]?.name ?? "";
    primeEditors();
    pushToast({
      tone: "success",
      title: "Provider 已删除",
      description: `${name} 及其相关配置已从当前配置中移除。`,
    });
  } catch (error) {
    handleError(error, "删除 Provider 失败");
  } finally {
    deletingProvider.value = false;
  }
}

async function submitAgent() {
  if (!agentDraft.name) {
    return false;
  }
  savingAgent.value = true;
  try {
    settings.value = await saveAgent(agentDraft.name, {
      label: agentDraft.label.trim(),
      provider: agentDraft.provider,
      model_name: agentDraft.model_name.trim(),
      temperature: parseOptionalNumber(agentDraft.temperature),
      reasoning_effort: emptyToUndefined(agentDraft.reasoning_effort),
    });
    invalidateConnectivity("agent", agentDraft.name);
    primeEditors();
    pushToast({
      tone: "success",
      title: "智能体配置已保存",
      description: `${agentDraft.name} 已写回后端配置文件。`,
    });
    return true;
  } catch (error) {
    handleError(error, "保存智能体失败");
    return false;
  } finally {
    savingAgent.value = false;
  }
}

async function submitEmbedding() {
  if (!embeddingDraft.name) {
    return false;
  }
  savingEmbedding.value = true;
  try {
    settings.value = await saveEmbeddingProfile(embeddingDraft.name, {
      label: embeddingDraft.label.trim(),
      provider: embeddingDraft.provider,
      model_name: embeddingDraft.model_name.trim(),
      dimensions: parseOptionalInteger(embeddingDraft.dimensions),
      batch_size: parseOptionalInteger(embeddingDraft.batch_size),
    });
    invalidateConnectivity("embedding_profile", embeddingDraft.name);
    primeEditors();
    pushToast({
      tone: "success",
      title: "嵌入模型已保存",
      description: `${embeddingDraft.name} 的向量配置已更新。`,
    });
    return true;
  } catch (error) {
    handleError(error, "保存嵌入模型失败");
    return false;
  } finally {
    savingEmbedding.value = false;
  }
}

// 中文注释：下面四个函数负责“新增智能体 / 新增嵌入模型”的进入与取消，
// 和 Provider 的新增流程保持同一套交互：先显示空白表单，保存成功后退出新增态。
function startCreatingAgent() {
  isCreatingAgent.value = true;
  editingAgentName.value = "";
  agentDraft.name = "";
  agentDraft.label = "";
  agentDraft.provider = providers.value[0]?.name ?? "";
  agentDraft.model_name = "";
  agentDraft.temperature = toInputString(settings.value?.defaults.llm.temperature ?? null);
  agentDraft.reasoning_effort = "none";
}

function cancelCreatingAgent() {
  isCreatingAgent.value = false;
  editingAgentName.value = agents.value[0]?.name ?? "";
}

function startCreatingEmbedding() {
  isCreatingEmbedding.value = true;
  editingEmbeddingName.value = "";
  embeddingDraft.name = "";
  embeddingDraft.label = "";
  embeddingDraft.provider = providers.value[0]?.name ?? "";
  embeddingDraft.model_name = "";
  embeddingDraft.dimensions = toInputString(settings.value?.defaults.embedding.dimensions ?? null);
  embeddingDraft.batch_size = toInputString(settings.value?.defaults.embedding.batch_size ?? null);
}

function cancelCreatingEmbedding() {
  isCreatingEmbedding.value = false;
  editingEmbeddingName.value = embeddings.value[0]?.name ?? "";
}

function selectAgentForEdit(name: string) {
  isCreatingAgent.value = false;
  editingAgentName.value = name;
}

function selectEmbeddingForEdit(name: string) {
  isCreatingEmbedding.value = false;
  editingEmbeddingName.value = name;
}

async function createAgent() {
  const name = agentDraft.name.trim();
  if (!name) {
    pushToast({
      tone: "error",
      title: "缺少参数",
      description: "请填写智能体名称（例如 default_agent）。",
    });
    return;
  }
  const saved = await submitAgent();
  if (!saved) {
    return;
  }
  isCreatingAgent.value = false;
  editingAgentName.value = name;
  primeEditors();
}

async function createEmbedding() {
  const name = embeddingDraft.name.trim();
  if (!name) {
    pushToast({
      tone: "error",
      title: "缺少参数",
      description: "请填写嵌入模型配置名称（例如 default_embedding）。",
    });
    return;
  }
  const saved = await submitEmbedding();
  if (!saved) {
    return;
  }
  isCreatingEmbedding.value = false;
  editingEmbeddingName.value = name;
  primeEditors();
}

async function fetchModels(providerName: string, toneTitle = "模型目录已同步") {
  if (!providerName) {
    return;
  }
  testingState[providerName] = true;
  try {
    // 中文注释：这里只同步模型目录，方便用户挑选模型名，不再把它当成真实连通性测试。
    const payload = await getProviderModels(providerName);
    providerModelsMap[providerName] = payload;
    if (payload.status === "available") {
      pushToast({
        tone: "success",
        title: toneTitle,
        description: `${providerName} 共返回 ${payload.model_count} 个可选模型。`,
      });
    } else {
      pushToast({
        tone: "info",
        title: "模型目录结果已返回",
        description: payload.message || `${providerName} 当前没有可用模型目录。`,
      });
    }
  } catch (error) {
    handleError(error, "获取模型目录失败");
  } finally {
    testingState[providerName] = false;
  }
}

async function runConnectivityTest(targetType: "agent" | "embedding_profile", name: string) {
  if (!name) {
    return;
  }
  const key = connectivityKey(targetType, name);
  connectivityTestingState[key] = true;
  try {
    const payload = await testModelConnectivity(targetType, name);
    connectivityMap[key] = payload;
    if (payload.status === "passed") {
      pushToast({
        tone: "success",
        title: "连通性测试通过",
        description: `${payload.name} · ${payload.model} · ${payload.message} · ${payload.latency_ms}ms`,
      });
      return;
    }
    if (payload.status === "not_configured") {
      pushToast({
        tone: "info",
        title: "模型配置尚未就绪",
        description: payload.message,
      });
      return;
    }
    pushToast({
      tone: "error",
      title: "连通性测试失败",
      description: payload.message,
    });
  } catch (error) {
    handleError(error, "执行模型连通性测试失败");
  } finally {
    connectivityTestingState[key] = false;
  }
}

async function testAgentConnection(name: string) {
  await runConnectivityTest("agent", name);
}

async function testEmbeddingConnection(name: string) {
  await runConnectivityTest("embedding_profile", name);
}

function providerStatusTone(provider: ProviderItem) {
  return provider.configured ? "success" : "warning";
}

function providerStatusLabel(provider: ProviderItem) {
  return provider.configured ? "已启用" : "待配置";
}

function modelCatalogStatus(providerName: string) {
  const payload = providerModelsMap[providerName];
  if (!payload) {
    return null;
  }
  if (payload.status === "available") {
    return { tone: "success" as const, label: `${payload.model_count} models` };
  }
  return { tone: "warning" as const, label: payload.status };
}

function agentSummary(agent: AgentItem) {
  return agent.description;
}

function embeddingSummary(item: EmbeddingProfileItem) {
  return `维度 ${item.dimensions ?? "默认"} · 批量 ${item.batch_size ?? "默认"} · Provider ${item.provider}`;
}

function connectivityLabel(targetType: "agent" | "embedding_profile", name: string) {
  const key = connectivityKey(targetType, name);
  if (connectivityTestingState[key]) {
    return "测试中...";
  }
  const payload = connectivityMap[key];
  if (!payload) {
    return "未测试";
  }
  if (payload.status === "passed") {
    return "已通过";
  }
  if (payload.status === "not_configured") {
    return "未配置";
  }
  return "未通过";
}

function connectivityKey(targetType: "agent" | "embedding_profile", name: string) {
  return `${targetType}:${name}`;
}

function invalidateConnectivity(targetType: "agent" | "embedding_profile", name: string) {
  const key = connectivityKey(targetType, name);
  delete connectivityMap[key];
  delete connectivityTestingState[key];
}

function invalidateProviderDerivedState(providerName: string) {
  delete providerModelsMap[providerName];
  delete testingState[providerName];
  for (const item of agents.value) {
    if (item.provider === providerName) {
      invalidateConnectivity("agent", item.name);
    }
  }
  for (const item of embeddings.value) {
    if (item.provider === providerName) {
      invalidateConnectivity("embedding_profile", item.name);
    }
  }
}

function resetTransientState() {
  clearRecord(providerModelsMap);
  clearRecord(testingState);
  clearRecord(connectivityMap);
  clearRecord(connectivityTestingState);
}

function clearRecord(record: Record<string, unknown>) {
  for (const key of Object.keys(record)) {
    delete record[key];
  }
}

function currentYear() {
  return new Date().getFullYear();
}

function toInputString(value: string | number | null) {
  return value === null || value === undefined ? "" : String(value);
}

function emptyToUndefined(value: string) {
  const normalized = value.trim();
  return normalized ? normalized : undefined;
}

function parseOptionalNumber(value: string) {
  const normalized = value.trim();
  if (!normalized) {
    return undefined;
  }
  const parsed = Number(normalized);
  if (Number.isNaN(parsed)) {
    throw new Error(`数值格式不正确: ${value}`);
  }
  return parsed;
}

function parseOptionalInteger(value: string) {
  const parsed = parseOptionalNumber(value);
  if (parsed === undefined) {
    return undefined;
  }
  if (!Number.isInteger(parsed)) {
    throw new Error(`需要整数值: ${value}`);
  }
  return parsed;
}

function prettyJson(value: Record<string, unknown>) {
  return JSON.stringify(value ?? {}, null, 2);
}

function parseJsonRecord(source: string, fieldName: string) {
  const normalized = source.trim();
  if (!normalized) {
    return {};
  }
  try {
    // 中文注释：Provider 扩展字段允许自由输入，但保存前必须收敛成 JSON 对象，避免写入脏值。
    const parsed = JSON.parse(normalized) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`${fieldName} 需要是 JSON 对象`);
    }
    return parsed as Record<string, unknown>;
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`${fieldName} 解析失败: ${error.message}`);
    }
    throw new Error(`${fieldName} 解析失败`);
  }
}

function handleError(error: unknown, title: string) {
  const description = error instanceof Error ? error.message : "未知错误";
  pushToast({
    tone: "error",
    title,
    description,
  });
}
</script>

<template>
  <section class="page-shell settings-page">
    <header class="hero-card">
      <div class="hero-copy">
        <span class="eyebrow">Runtime Configuration</span>
        <h1>系统配置工作台</h1>
        <p>
          用一个干净的控制台统一管理模型上游、智能体与嵌入配置。
          每次保存都会即时落盘，下一次请求直接生效。
        </p>
      </div>
      <div class="hero-actions">
        <button class="button secondary" type="button" :disabled="refreshing" @click="refreshSettings">
          <RefreshCw :size="16" :class="{ spinning: refreshing }" />
          重新加载
        </button>
        <a class="button ghost-link" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
          API 文档
          <ArrowUpRight :size="16" />
        </a>
      </div>
    </header>

    <section v-if="loading" class="state-card">
      <LoaderCircle class="spinning" :size="20" />
      <span>正在读取后端配置与可用能力...</span>
    </section>

    <template v-else-if="settings">
      <section class="metrics-grid">
        <article v-for="card in healthCards" :key="card.label" class="metric-card">
          <div class="metric-icon"><component :is="card.icon" :size="20" /></div>
          <div class="metric-copy">
            <div class="metric-value-row"><strong>{{ card.value }}</strong><span>{{ card.label }}</span></div>
            <p>{{ card.detail }}</p>
          </div>
        </article>
      </section>

      <section class="panel-section">
        <div class="section-heading">
          <div>
            <span class="eyebrow">01 Provider Workspace</span>
            <h2>模型提供商配置</h2>
            <p>Provider 类型来自后端接口，当前参数直接回填自配置文件。</p>
          </div>
          <div class="heading-meta">
            <StatusPill
              :tone="settings.requires_restart ? 'warning' : 'success'"
              :label="settings.requires_restart ? '需要重启' : '热生效'"
            />
            <StatusPill tone="neutral" :label="`${providerCompletion}% 完整度`" />
          </div>
        </div>

        <div class="provider-stage">
          <aside class="provider-rail">
            <div class="provider-list-header">
              <strong>Provider 列表</strong>
              <button
                class="button secondary provider-add-button"
                type="button"
                :aria-pressed="isCreatingProvider"
                @click="startCreatingProvider"
              >
                <Plus :size="15" />
                新增 Provider
              </button>
            </div>

            <button
              v-for="provider in providers"
              :key="provider.name"
              class="provider-tab"
              :class="{ active: provider.name === selectedProviderName }"
              type="button"
              @click="selectProvider(provider.name)"
            >
              <div class="provider-tab-main">
                <strong>{{ provider.label }}</strong>
                <span>{{ provider.name }}</span>
              </div>
              <StatusPill
                :tone="providerStatusTone(provider)"
                :label="providerStatusLabel(provider)"
              />
            </button>
          </aside>

          <div v-if="isCreatingProvider || selectedProvider" class="provider-editor-card">
            <div class="provider-header-row">
              <div>
                <template v-if="isCreatingProvider">
                  <h3>新增 Provider</h3>
                  <p>在右侧完成初始化，保存后会加入 Provider 列表。</p>
                </template>
                <template v-else>
                  <h3>{{ selectedProvider?.label }}</h3>
                  <p>{{ selectedProvider?.name }} · {{ selectedProvider?.provider_type }}</p>
                </template>
              </div>
              <div class="provider-toolbar">
                <template v-if="isCreatingProvider">
                  <button class="button secondary" type="button" @click="cancelCreatingProvider">
                    取消
                  </button>
                  <button class="button primary" type="button" :disabled="savingProvider" @click="createProvider">
                    <Save :size="16" />
                    创建 Provider
                  </button>
                </template>
                <template v-else>
                  <button
                    class="button secondary"
                    type="button"
                    :disabled="testingState[selectedProvider?.name || '']"
                    @click="fetchModels(selectedProvider?.name || '')"
                  >
                    <SearchCheck :size="16" />
                    同步模型目录
                  </button>
                  <button class="button primary" type="button" :disabled="savingProvider" @click="submitProvider">
                    <Save :size="16" />
                    保存 Provider
                  </button>
                  <button
                    class="button danger"
                    type="button"
                    :disabled="deletingProvider"
                    @click="deleteCurrentProvider"
                  >
                    <Trash2 :size="16" />
                    删除 Provider
                  </button>
                </template>
              </div>
            </div>

            <div class="form-grid provider-form-grid">
              <label v-if="isCreatingProvider" class="field-group">
                <span>Provider ID</span>
                <input
                  v-model="creatingProvider.name"
                  class="field"
                  type="text"
                  placeholder="例如 openai_main"
                />
              </label>

              <label class="field-group">
                <span>Provider 类型</span>
                <select v-model="providerBackend" class="field">
                  <option
                    v-for="item in providerTypes"
                    :key="item.name"
                    :value="item.name"
                  >
                    {{ item.label }}
                  </option>
                </select>
              </label>

              <label class="field-group">
                <span>API Key 环境变量</span>
                <input
                  v-model="providerDraft.api_key_env"
                  class="field"
                  type="text"
                  placeholder="例如 OPENAI_API_KEY"
                />
              </label>

              <label class="field-group field-group-wide">
                <span>API Key</span>
                <div class="field-inline">
                  <input
                    v-model="providerDraft.api_key"
                    class="field"
                    :type="showApiKey ? 'text' : 'password'"
                    placeholder="输入可用于该 Provider 的密钥"
                  />
                  <button class="button secondary compact" type="button" @click="showApiKey = !showApiKey">
                    {{ showApiKey ? "隐藏" : "显示" }}
                  </button>
                </div>
              </label>

              <label class="field-group field-group-wide">
                <span>API Base URL</span>
                <input
                  v-model="providerDraft.api_base"
                  class="field"
                  type="text"
                  :placeholder="selectedProviderType?.default_api_base || 'https://api.example.com/v1'"
                />
              </label>

              <label class="field-group field-group-wide">
                <span>Extra Headers</span>
                <textarea
                  v-model="providerDraft.extra_headers"
                  class="field textarea"
                  rows="7"
                  spellcheck="false"
                />
              </label>

              <label class="field-group field-group-wide">
                <span>Extra Body</span>
                <textarea
                  v-model="providerDraft.extra_body"
                  class="field textarea"
                  rows="7"
                  spellcheck="false"
                />
              </label>
            </div>

            <template v-if="!isCreatingProvider && selectedProvider">
              <div class="provider-footer">
                <div class="mini-note">
                  <span class="eyebrow">Model Catalog</span>
                  <p>
                    目录抓取只用于辅助选择模型名，不能代表当前智能体或嵌入配置已经真实可调用。
                  </p>
                </div>
                <StatusPill
                  v-if="modelCatalogStatus(selectedProvider.name)"
                  :tone="modelCatalogStatus(selectedProvider.name)?.tone"
                  :label="modelCatalogStatus(selectedProvider.name)?.label || ''"
                />
              </div>

              <div class="model-catalog">
                <div v-if="activeProviderModels.length" class="catalog-grid">
                  <article
                    v-for="model in activeProviderModels.slice(0, 8)"
                    :key="model.id"
                    class="catalog-item"
                  >
                    <strong>{{ model.label }}</strong>
                    <span>{{ model.owned_by || "upstream" }}</span>
                  </article>
                </div>
                <div v-else class="empty-line">
                  <span>尚未同步模型目录，点击“同步模型目录”即可拉取。</span>
                </div>
              </div>
            </template>
          </div>
        </div>
      </section>

      <section class="panel-section">
        <div class="section-heading">
          <div>
            <span class="eyebrow">02 Agent Matrix</span>
            <h2>智能体配置</h2>
            <p>以表格管理大量智能体，并为每个智能体提供独立的连通性测试入口。</p>
          </div>
          <div class="heading-meta">
            <StatusPill tone="neutral" :label="`${agents.length} agents`" />
            <button class="button secondary" type="button" @click="startCreatingAgent">
              <Plus :size="15" />
              新增智能体
            </button>
          </div>
        </div>

        <div class="table-card">
          <table class="data-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>依赖 Provider</th>
                <th>依赖模型</th>
                <th>简要描述</th>
                <th>连通性</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="agent in agents" :key="agent.name">
                <td>
                  <div class="primary-cell">
                    <strong>{{ agent.label }}</strong>
                    <span>{{ agent.name }}</span>
                  </div>
                </td>
                <td>{{ agent.provider }}</td>
                <td>{{ agent.model_name }}</td>
                <td>{{ agentSummary(agent) }}</td>
                <td>
                  <button
                    class="button tertiary compact"
                    type="button"
                    :disabled="connectivityTestingState[connectivityKey('agent', agent.name)]"
                    @click="testAgentConnection(agent.name)"
                  >
                    {{ connectivityLabel('agent', agent.name) }}
                  </button>
                </td>
                <td class="align-right">
                  <button class="button secondary compact" type="button" @click="selectAgentForEdit(agent.name)">
                    编辑
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-if="isCreatingAgent || editingAgent" class="editor-card">
          <div class="editor-head">
            <div>
              <h3>{{ isCreatingAgent ? "新增智能体" : "编辑智能体" }}</h3>
              <p v-if="isCreatingAgent">填写后保存，将创建新的智能体配置。</p>
              <p v-else>{{ editingAgent?.name }} 的运行时模型配置</p>
            </div>
            <div class="provider-toolbar">
              <template v-if="isCreatingAgent">
                <button class="button secondary" type="button" @click="cancelCreatingAgent">
                  取消
                </button>
                <button class="button primary" type="button" :disabled="savingAgent" @click="createAgent">
                  <Save :size="16" />
                  创建智能体
                </button>
              </template>
              <template v-else>
                <button class="button primary" type="button" :disabled="savingAgent" @click="submitAgent">
                  <Save :size="16" />
                  保存智能体
                </button>
              </template>
            </div>
          </div>

          <div class="form-grid">
            <label v-if="isCreatingAgent" class="field-group">
              <span>智能体名称</span>
              <input
                v-model="agentDraft.name"
                class="field"
                type="text"
                placeholder="例如 default_agent"
              />
            </label>
            <label class="field-group">
              <span>显示名称</span>
              <input v-model="agentDraft.label" class="field" type="text" />
            </label>
            <label class="field-group">
              <span>Provider</span>
              <select v-model="agentDraft.provider" class="field">
                <option v-for="provider in providers" :key="provider.name" :value="provider.name">
                  {{ provider.label }}
                </option>
              </select>
            </label>
            <label class="field-group field-group-wide">
              <span>模型名称</span>
              <input
                v-model="agentDraft.model_name"
                class="field"
                list="agent-model-options"
                type="text"
                placeholder="输入或选择模型"
              />
              <datalist id="agent-model-options">
                <option
                  v-for="model in agentProviderModels"
                  :key="model.id"
                  :value="model.id"
                />
              </datalist>
            </label>
            <label class="field-group">
              <span>Temperature</span>
              <input v-model="agentDraft.temperature" class="field" type="number" step="0.1" />
            </label>
            <label class="field-group">
              <span>Reasoning Effort</span>
              <select v-model="agentDraft.reasoning_effort" class="field">
                <option value="none">none</option>
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="high">high</option>
              </select>
            </label>
          </div>

          <div class="editor-actions">
            <button
              class="button secondary"
              type="button"
              :disabled="testingState[agentDraft.provider]"
              @click="fetchModels(agentDraft.provider)"
            >
              <RefreshCw :size="16" />
              刷新该 Provider 模型
            </button>
          </div>
        </div>
      </section>

      <section class="panel-section">
        <div class="section-heading">
          <div>
            <span class="eyebrow">03 Embedding Profiles</span>
            <h2>嵌入模型配置</h2>
            <p>向量化配置与智能体独立管理，适合检索、索引和召回链路。</p>
          </div>
          <div class="heading-meta">
            <StatusPill tone="neutral" :label="`${embeddings.length} profiles`" />
            <button class="button secondary" type="button" @click="startCreatingEmbedding">
              <Plus :size="15" />
              新增嵌入模型
            </button>
          </div>
        </div>

        <div class="table-card">
          <table class="data-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>依赖 Provider</th>
                <th>依赖模型</th>
                <th>简要描述</th>
                <th>连通性</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in embeddings" :key="item.name">
                <td>
                  <div class="primary-cell">
                    <strong>{{ item.label }}</strong>
                    <span>{{ item.name }}</span>
                  </div>
                </td>
                <td>{{ item.provider }}</td>
                <td>{{ item.model_name }}</td>
                <td>{{ embeddingSummary(item) }}</td>
                <td>
                  <button
                    class="button tertiary compact"
                    type="button"
                    :disabled="connectivityTestingState[connectivityKey('embedding_profile', item.name)]"
                    @click="testEmbeddingConnection(item.name)"
                  >
                    {{ connectivityLabel('embedding_profile', item.name) }}
                  </button>
                </td>
                <td class="align-right">
                  <button class="button secondary compact" type="button" @click="selectEmbeddingForEdit(item.name)">
                    编辑
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-if="isCreatingEmbedding || editingEmbedding" class="editor-card">
          <div class="editor-head">
            <div>
              <h3>{{ isCreatingEmbedding ? "新增嵌入配置" : "编辑嵌入配置" }}</h3>
              <p v-if="isCreatingEmbedding">填写后保存，将创建新的嵌入模型配置。</p>
              <p v-else>{{ editingEmbedding?.name }} 的向量模型参数</p>
            </div>
            <div class="provider-toolbar">
              <template v-if="isCreatingEmbedding">
                <button class="button secondary" type="button" @click="cancelCreatingEmbedding">
                  取消
                </button>
                <button class="button primary" type="button" :disabled="savingEmbedding" @click="createEmbedding">
                  <Save :size="16" />
                  创建嵌入配置
                </button>
              </template>
              <template v-else>
                <button class="button primary" type="button" :disabled="savingEmbedding" @click="submitEmbedding">
                  <Save :size="16" />
                  保存嵌入配置
                </button>
              </template>
            </div>
          </div>

          <div class="form-grid">
            <label v-if="isCreatingEmbedding" class="field-group">
              <span>配置名称</span>
              <input
                v-model="embeddingDraft.name"
                class="field"
                type="text"
                placeholder="例如 default_embedding"
              />
            </label>
            <label class="field-group">
              <span>显示名称</span>
              <input v-model="embeddingDraft.label" class="field" type="text" />
            </label>
            <label class="field-group">
              <span>Provider</span>
              <select v-model="embeddingDraft.provider" class="field">
                <option v-for="provider in providers" :key="provider.name" :value="provider.name">
                  {{ provider.label }}
                </option>
              </select>
            </label>
            <label class="field-group field-group-wide">
              <span>模型名称</span>
              <input
                v-model="embeddingDraft.model_name"
                class="field"
                list="embedding-model-options"
                type="text"
                placeholder="输入或选择嵌入模型"
              />
              <datalist id="embedding-model-options">
                <option
                  v-for="model in embeddingProviderModels"
                  :key="model.id"
                  :value="model.id"
                />
              </datalist>
            </label>
            <label class="field-group">
              <span>维度</span>
              <input v-model="embeddingDraft.dimensions" class="field" type="number" step="1" />
            </label>
            <label class="field-group">
              <span>Batch Size</span>
              <input v-model="embeddingDraft.batch_size" class="field" type="number" step="1" />
            </label>
          </div>

          <div class="editor-actions">
            <button
              class="button secondary"
              type="button"
              :disabled="testingState[embeddingDraft.provider]"
              @click="fetchModels(embeddingDraft.provider)"
            >
              <RefreshCw :size="16" />
              刷新该 Provider 模型
            </button>
          </div>
        </div>
      </section>
    </template>

    <footer class="page-footer">
      <span>Minimal control surface for papers agents runtime.</span>
      <span>{{ currentYear() }}</span>
    </footer>
  </section>
</template>
