<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { CalendarDays, Download, FileJson, LoaderCircle, PackageOpen, X } from "lucide-vue-next";

import { downloadSessionTrace, fetchSessionThread, fetchSessionTraces } from "../../api/sessions";
import { SessionStreamAggregator } from "../../lib/session-stream-aggregator";
import type {
  SessionThread,
  SessionTraceSummary,
  UIRuntimeTimelineEvent,
} from "../../types/sessions";
import RuntimeEventTree from "./RuntimeEventTree.vue";

const props = defineProps<{
  sessionKey: string;
}>();

const emit = defineEmits<{
  close: [];
}>();

const traces = ref<SessionTraceSummary[]>([]);
const thread = ref<SessionThread | null>(null);
const selectedTurnId = ref("");
const loading = ref(true);
const downloading = ref(false);
const errorMessage = ref("");
const activeTab = ref<"events" | "artifacts">("events");
let loadGeneration = 0;

const selectedTrace = computed(() => {
  return traces.value.find((trace) => trace.turn_id === selectedTurnId.value) ?? null;
});

const selectedEvents = computed(() => {
  return (thread.value?.events ?? []).filter((event) => event.metadata.turn_id === selectedTurnId.value);
});

const selectedMessages = computed(() => {
  return (thread.value?.messages ?? []).filter((message) => message.turn_id === selectedTurnId.value);
});

const selectedArtifacts = computed(() => {
  return (thread.value?.artifacts ?? []).filter((artifact) => artifact.metadata.turn_id === selectedTurnId.value);
});

const runtimeEvents = computed(() => {
  if (!thread.value || !selectedTrace.value) return [];
  // 中文说明：抽屉使用独立聚合器，只重建选中的历史 turn，不影响外部实时时间线。
  const aggregator = new SessionStreamAggregator();
  aggregator.hydrate({
    ...thread.value,
    status: selectedTrace.value.status,
    events: selectedEvents.value,
    messages: selectedMessages.value,
    artifacts: selectedArtifacts.value,
    has_pending_tool_calls: false,
    run_started_at: null,
  });
  return withoutResumeActions(aggregator.snapshot().runtimeEvents);
});

watch(
  () => props.sessionKey,
  () => void loadTraceData(),
  { immediate: true },
);

onMounted(() => window.addEventListener("keydown", handleKeydown));
onBeforeUnmount(() => window.removeEventListener("keydown", handleKeydown));

async function loadTraceData() {
  const generation = ++loadGeneration;
  loading.value = true;
  errorMessage.value = "";
  thread.value = null;
  traces.value = [];
  try {
    const [tracePayload, threadPayload] = await Promise.all([
      fetchSessionTraces(props.sessionKey),
      fetchSessionThread(props.sessionKey),
    ]);
    if (generation !== loadGeneration) return;
    traces.value = tracePayload.traces;
    thread.value = threadPayload;
    selectedTurnId.value = tracePayload.traces.some((trace) => trace.turn_id === selectedTurnId.value)
      ? selectedTurnId.value
      : tracePayload.traces[0]?.turn_id ?? "";
  } catch (error) {
    if (generation !== loadGeneration) return;
    errorMessage.value = error instanceof Error ? error.message : "轨迹加载失败";
  } finally {
    if (generation === loadGeneration) loading.value = false;
  }
}

async function downloadSelectedTrace() {
  if (!selectedTurnId.value || downloading.value) return;
  downloading.value = true;
  errorMessage.value = "";
  try {
    await downloadSessionTrace(props.sessionKey, selectedTurnId.value);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "轨迹下载失败";
  } finally {
    downloading.value = false;
  }
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") emit("close");
}

function withoutResumeActions(events: UIRuntimeTimelineEvent[]): UIRuntimeTimelineEvent[] {
  return events.map((event) => ({
    ...event,
    resumeAvailable: false,
    children: withoutResumeActions(event.children),
  }));
}

function artifactUrl(artifactId: string) {
  return `/api/sessions/${encodeURIComponent(props.sessionKey)}/artifacts/${encodeURIComponent(artifactId)}`;
}

function formatTime(value: string | null) {
  if (!value) return "未知";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function formatSize(size: number) {
  if (!size) return "0 B";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    completed: "已完成",
    running: "运行中",
    failed: "失败",
    cancelled: "已停止",
    interrupted: "已中断",
    unknown: "状态未知",
  };
  return labels[status] ?? status;
}

function formatRawEvent(value: unknown) {
  return JSON.stringify(value, null, 2);
}
</script>

<template>
  <div class="trace-drawer-overlay" @click.self="emit('close')">
    <aside class="trace-drawer" role="dialog" aria-modal="true" aria-label="运行轨迹">
      <header class="trace-drawer-header">
        <div>
          <span class="trace-drawer-eyebrow">Run Trace</span>
          <h2>运行轨迹</h2>
          <p>查看历史执行树和原始事件，或下载完整轨迹包。</p>
        </div>
        <button class="trace-icon-button" type="button" aria-label="关闭轨迹" @click="emit('close')">
          <X :size="20" />
        </button>
      </header>

      <div v-if="traces.length" class="trace-drawer-toolbar">
        <label class="trace-turn-picker">
          <span>运行回合</span>
          <select v-model="selectedTurnId">
            <option v-for="trace in traces" :key="trace.turn_id" :value="trace.turn_id">
              {{ trace.turn_id }} · {{ statusLabel(trace.status) }}
            </option>
          </select>
        </label>
        <button
          class="button compact trace-download-button"
          type="button"
          :disabled="!selectedTurnId || downloading"
          @click="downloadSelectedTrace"
        >
          <LoaderCircle v-if="downloading" class="spinning" :size="15" />
          <Download v-else :size="15" />
          {{ downloading ? "正在打包" : "下载轨迹" }}
        </button>
      </div>

      <div v-if="loading" class="trace-drawer-empty">
        <LoaderCircle class="spinning" :size="22" />
        <span>正在加载历史轨迹…</span>
      </div>
      <div v-else-if="errorMessage && !traces.length" class="trace-drawer-empty trace-drawer-error">
        <strong>轨迹不可用</strong>
        <span>{{ errorMessage }}</span>
        <button class="button compact" type="button" @click="loadTraceData">重新加载</button>
      </div>
      <div v-else-if="!traces.length" class="trace-drawer-empty">
        <FileJson :size="28" />
        <strong>还没有可查看的运行轨迹</strong>
        <span>执行一次任务后，事件和产物会显示在这里。</span>
      </div>

      <template v-else-if="selectedTrace">
        <div v-if="errorMessage" class="trace-action-error">{{ errorMessage }}</div>
        <section class="trace-summary" aria-label="轨迹摘要">
          <div><span>状态</span><strong>{{ statusLabel(selectedTrace.status) }}</strong></div>
          <div><span>开始</span><strong>{{ formatTime(selectedTrace.started_at) }}</strong></div>
          <div><span>事件</span><strong>{{ selectedTrace.event_count }}</strong></div>
          <div><span>产物</span><strong>{{ selectedTrace.artifact_count }}</strong></div>
        </section>

        <div class="trace-drawer-body">
          <section class="trace-tree-panel">
            <div class="trace-panel-heading">
              <h3>执行树</h3>
              <span>{{ selectedEvents.length }} 条原始事件</span>
            </div>
            <RuntimeEventTree v-if="runtimeEvents.length" :events="runtimeEvents" />
            <div v-else class="trace-panel-empty">该回合没有节点执行树，请查看原始事件。</div>
          </section>

          <section class="trace-detail-panel">
            <div class="trace-detail-tabs" role="tablist">
              <button
                type="button"
                :data-active="activeTab === 'events'"
                @click="activeTab = 'events'"
              >
                <FileJson :size="14" />原始事件
              </button>
              <button
                type="button"
                :data-active="activeTab === 'artifacts'"
                @click="activeTab = 'artifacts'"
              >
                <PackageOpen :size="14" />产物 {{ selectedArtifacts.length }}
              </button>
            </div>

            <div v-if="activeTab === 'events'" class="trace-raw-events">
              <details v-for="event in selectedEvents" :key="event.id" class="trace-raw-event">
                <summary>
                  <span>#{{ event.seq_no }} {{ event.event_type }}</span>
                  <time>{{ formatTime(event.created_at) }}</time>
                </summary>
                <pre>{{ formatRawEvent(event) }}</pre>
              </details>
              <div v-if="!selectedEvents.length" class="trace-panel-empty">该回合没有事件记录。</div>
            </div>

            <div v-else class="trace-artifact-list">
              <a
                v-for="artifact in selectedArtifacts"
                :key="artifact.id"
                :href="artifactUrl(artifact.id)"
                target="_blank"
                rel="noreferrer"
                class="trace-artifact-item"
              >
                <PackageOpen :size="17" />
                <span><strong>{{ artifact.name }}</strong><small>{{ artifact.artifact_type }} · {{ formatSize(artifact.size) }}</small></span>
                <Download :size="15" />
              </a>
              <div v-if="!selectedArtifacts.length" class="trace-panel-empty">该回合没有产物记录。</div>
            </div>
          </section>
        </div>
        <footer class="trace-drawer-footer">
          <CalendarDays :size="13" />
          <span>数据来自现有会话事件、消息与产物；zip 导出会移除密钥和本地路径。</span>
        </footer>
      </template>
    </aside>
  </div>
</template>

<style scoped>
.trace-drawer-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  justify-content: flex-end;
  background: rgba(15, 23, 38, 0.32);
  backdrop-filter: blur(3px);
}

.trace-drawer {
  display: flex;
  flex-direction: column;
  width: min(1120px, calc(100vw - 72px));
  height: 100%;
  min-height: 0;
  background: var(--bg);
  box-shadow: -24px 0 80px rgba(15, 23, 38, 0.16);
}

.trace-drawer-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  padding: 24px 26px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--surface-strong);
}

.trace-drawer-header h2,
.trace-drawer-header p {
  margin: 0;
}

.trace-drawer-header h2 {
  margin-top: 3px;
  font-size: 1.35rem;
}

.trace-drawer-header p {
  margin-top: 6px;
  color: var(--text-soft);
  font-size: 0.82rem;
}

.trace-drawer-eyebrow {
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.trace-icon-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface-muted);
  color: var(--text-soft);
}

.trace-drawer-toolbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 26px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}

.trace-turn-picker {
  display: grid;
  flex: 1;
  gap: 5px;
  max-width: 520px;
  color: var(--text-soft);
  font-size: 0.7rem;
}

.trace-turn-picker select {
  width: 100%;
  min-width: 0;
  padding: 9px 12px;
  border: 1px solid var(--border-strong);
  border-radius: 10px;
  background: var(--surface-strong);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 0.75rem;
}

.trace-download-button {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  white-space: nowrap;
}

.trace-summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1px;
  margin: 14px 26px 0;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--border);
}

.trace-summary > div {
  display: grid;
  gap: 4px;
  padding: 10px 13px;
  background: var(--surface-strong);
}

.trace-summary span {
  color: var(--text-faint);
  font-size: 0.65rem;
}

.trace-summary strong {
  overflow: hidden;
  font-size: 0.76rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trace-drawer-body {
  display: grid;
  grid-template-columns: minmax(310px, 0.9fr) minmax(360px, 1.1fr);
  flex: 1;
  gap: 14px;
  min-height: 0;
  padding: 14px 26px;
}

.trace-tree-panel,
.trace-detail-panel {
  min-width: 0;
  min-height: 0;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface-strong);
}

.trace-tree-panel {
  padding: 14px;
}

.trace-panel-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.trace-panel-heading h3 {
  margin: 0;
  font-size: 0.85rem;
}

.trace-panel-heading span {
  color: var(--text-faint);
  font-size: 0.65rem;
}

.trace-detail-tabs {
  position: sticky;
  top: 0;
  z-index: 1;
  display: flex;
  gap: 4px;
  padding: 9px;
  border-bottom: 1px solid var(--border);
  background: var(--surface-strong);
}

.trace-detail-tabs button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 10px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--text-soft);
  font-size: 0.7rem;
}

.trace-detail-tabs button[data-active="true"] {
  background: var(--accent-soft);
  color: var(--accent);
}

.trace-raw-events,
.trace-artifact-list {
  display: grid;
  gap: 8px;
  padding: 10px;
}

.trace-raw-event {
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--surface-muted);
}

.trace-raw-event summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 9px 10px;
  cursor: pointer;
  color: var(--text-soft);
  font-family: var(--font-mono);
  font-size: 0.66rem;
}

.trace-raw-event time {
  color: var(--text-faint);
  white-space: nowrap;
}

.trace-raw-event pre {
  max-height: 420px;
  margin: 0;
  overflow: auto;
  padding: 12px;
  border-top: 1px solid var(--border);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 0.65rem;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.trace-artifact-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px;
  border: 1px solid var(--border);
  border-radius: 10px;
  color: var(--text-soft);
}

.trace-artifact-item:hover {
  border-color: rgba(31, 143, 181, 0.3);
  background: var(--accent-soft);
}

.trace-artifact-item > span {
  display: grid;
  flex: 1;
  min-width: 0;
  gap: 3px;
}

.trace-artifact-item strong {
  overflow: hidden;
  color: var(--text);
  font-size: 0.74rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trace-artifact-item small {
  color: var(--text-faint);
  font-size: 0.63rem;
}

.trace-panel-empty,
.trace-drawer-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-faint);
  text-align: center;
}

.trace-panel-empty {
  min-height: 140px;
  padding: 24px;
  font-size: 0.75rem;
}

.trace-drawer-empty {
  flex: 1;
  flex-direction: column;
  gap: 10px;
  padding: 40px;
  font-size: 0.82rem;
}

.trace-drawer-empty strong {
  color: var(--text);
}

.trace-drawer-error,
.trace-action-error {
  color: var(--danger);
}

.trace-action-error {
  margin: 12px 26px 0;
  padding: 9px 12px;
  border-radius: 9px;
  background: var(--danger-soft);
  font-size: 0.75rem;
}

.trace-drawer-footer {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 10px 26px 14px;
  color: var(--text-faint);
  font-size: 0.64rem;
}

@media (max-width: 900px) {
  .trace-drawer {
    width: 100vw;
  }

  .trace-drawer-body {
    grid-template-columns: 1fr;
    overflow: auto;
  }

  .trace-tree-panel,
  .trace-detail-panel {
    overflow: visible;
  }
}

@media (max-width: 620px) {
  .trace-drawer-header,
  .trace-drawer-toolbar {
    padding-inline: 16px;
  }

  .trace-drawer-toolbar {
    align-items: stretch;
    flex-direction: column;
  }

  .trace-turn-picker {
    max-width: none;
  }

  .trace-summary {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin-inline: 16px;
  }

  .trace-drawer-body {
    padding-inline: 16px;
  }
}
</style>
