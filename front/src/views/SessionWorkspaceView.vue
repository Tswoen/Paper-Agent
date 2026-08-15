<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import {
  CalendarDays,
  CheckCircle2,
  FileDown,
  FileSearch,
  Gauge,
  LoaderCircle,
  PackageOpen,
  Timer,
  Upload,
} from "lucide-vue-next";

import { ApiRequestError, cancelSessionRun, createSession, fetchSessionThread, startSessionRun, subscribeSessionRun } from "../api/sessions";
import SessionComposer from "../components/session/SessionComposer.vue";
import SessionTimeline from "../components/session/SessionTimeline.vue";
import StatusPill from "../components/StatusPill.vue";
import { createReadableId } from "../lib/random-id";
import { SessionStreamAggregator } from "../lib/session-stream-aggregator";
import { pushToast } from "../stores/notifications";
import type {
  SessionRuntimeEvent,
  SessionConstraints,
  SessionSummary,
  SessionThread,
  SessionTimelineSnapshot,
} from "../types/sessions";

const props = defineProps<{
  sessions: SessionSummary[];
  selectedKey: string;
  creatingSession: boolean;
}>();

const emit = defineEmits<{
  "update:selectedKey": [sessionKey: string];
  refreshSessions: [];
}>();

const selectedSessionKey = ref("");
const selectedTitle = ref("新的研究主题");
const selectedRunStartedAt = ref<string | null>(null);
const draft = ref("");
const constraints = ref<SessionConstraints>({});
const threadLoading = ref(false);
const sending = ref(false);
const cancelling = ref(false);
const activeRunId = ref<string | null>(null);
const streamSource = ref<EventSource | null>(null);
const manualClose = ref(false);
const timelineSnapshot = ref<SessionTimelineSnapshot | null>(null);

const aggregator = new SessionStreamAggregator();

const selectedSummary = computed(() => props.sessions.find((session) => session.key === selectedSessionKey.value));
const currentStatus = computed(() => timelineSnapshot.value?.status ?? selectedSummary.value?.status ?? "created");
const isRunning = computed(() => timelineSnapshot.value?.isStreaming ?? false);
const isBlankWorkspace = computed(() => !selectedSessionKey.value && !timelineSnapshot.value);

const shouldShowTimeline = computed(() => {
  return Boolean(
    sending.value ||
      isRunning.value ||
      selectedRunStartedAt.value ||
      selectedSummary.value?.run_started_at ||
      hasTimelineContent(timelineSnapshot.value),
  );
});

/**
 * 中文注释：欢迎页只在“还没有开始执行”的时候展示。
 * 这里不能只看有没有 selectedSessionKey，因为点击左侧新建会话后，后端已经有了会话编号，
 * 但用户还没有输入研究主题，所以页面仍然应该显示中间的大输入框。
 */
const shouldShowWelcomeComposer = computed(() => !threadLoading.value && !shouldShowTimeline.value);

const statusText = computed(() => {
  if (cancelling.value || currentStatus.value === "cancel_requested") {
    return "正在停止";
  }
  if (sending.value) {
    return "正在连接";
  }
  if (isRunning.value) {
    return "运行中";
  }
  if (isBlankWorkspace.value) {
    return "准备输入";
  }
  return "准备就绪";
});

const compactUpdatedAt = computed(() => {
  const value = selectedSummary.value?.updated_at;
  if (!value) {
    return "等待开始";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
});

/**
 * 右侧统计栏直接从当前时间线计算数据，避免再增加一套后端统计接口。
 * 事件数量、完成数量和 token 总数都能从已经加载的会话快照中得到。
 */
const timelineStats = computed(() => {
  const rootEvents = timelineSnapshot.value?.runtimeEvents ?? [];
  const events = flattenRuntimeEvents(rootEvents);
  // 右侧的“步骤”对应时间线最外层的节点，内部工具调用不单独算一步。
  const completed = rootEvents.filter((event) => event.status === "completed").length;
  // 父节点的 token 是子节点汇总值，统计时只取叶子节点，避免重复相加。
  const tokenEvents = events.filter((event) => event.children.length === 0);
  const inputTokens = tokenEvents.reduce((total, event) => total + event.inputTokens, 0);
  const outputTokens = tokenEvents.reduce((total, event) => total + event.outputTokens, 0);
  const startedAt = selectedRunStartedAt.value ?? selectedSummary.value?.run_started_at;
  const endAt = currentStatus.value === "running" ? undefined : selectedSummary.value?.updated_at;

  return {
    total: rootEvents.length,
    completed,
    inputTokens,
    outputTokens,
    duration: formatDuration(startedAt, endAt),
  };
});

const taskStatusLabel = computed(() => {
  if (currentStatus.value === "completed") return "已完成";
  if (currentStatus.value === "running") return "执行中";
  if (currentStatus.value === "cancel_requested") return "正在停止";
  if (currentStatus.value === "cancelled") return "已停止";
  if (currentStatus.value === "failed") return "执行失败";
  return "等待开始";
});

/**
 * 右侧只展示工作流最后生成的论文，检索结果和阅读笔记等中间文件不放在这里。
 * 后端用 final_review 标记最终论文，这样会话里保存再多其他产物，页面也不会把它们混在一起。
 */
const finalArtifacts = computed(() => {
  const artifacts = timelineSnapshot.value?.artifacts ?? [];
  return artifacts.filter((artifact) => artifact.artifact_type === "final_review");
});

/** 将树形事件展开成一维数组，只用于统计，不会改变时间线的显示结构。 */
function flattenRuntimeEvents(
  events: SessionTimelineSnapshot["runtimeEvents"],
): SessionTimelineSnapshot["runtimeEvents"] {
  return events.flatMap((event) => [event, ...flattenRuntimeEvents(event.children)]);
}

function formatDuration(start: string | null | undefined, end: string | null | undefined) {
  if (!start) return "--";
  const startTime = new Date(start).getTime();
  const endTime = end ? new Date(end).getTime() : Date.now();
  if (!Number.isFinite(startTime) || !Number.isFinite(endTime) || endTime < startTime) return "--";
  const seconds = Math.max(1, Math.round((endTime - startTime) / 1000));
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return minutes > 0 ? `${minutes}分${remainingSeconds}秒` : `${remainingSeconds}秒`;
}

function formatNumber(value: number) {
  return value.toLocaleString("zh-CN");
}

function formatArtifactSize(size: number) {
  if (!size) return "文件大小未知";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

/** 拼接产物文件的下载/预览地址，会话还没确定时返回 "#"，避免生成无效链接。 */
function artifactUrlFor(artifactId: string) {
  if (!selectedSessionKey.value || !artifactId) return "#";
  return `/api/sessions/${encodeURIComponent(selectedSessionKey.value)}/artifacts/${encodeURIComponent(artifactId)}`;
}

/** 取当前会话最后一条非空的助手回复，它就是工作流生成的完整 Markdown 论文。 */
const finalResultMarkdown = computed(() => {
  const messages = timelineSnapshot.value?.messages ?? [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role === "assistant" && message.content.trim()) {
      return message.content;
    }
  }
  return "";
});

watch(
  () => props.selectedKey,
  async (sessionKey) => {
    if (!sessionKey) {
      resetToBlankWorkspace();
      return;
    }
    await selectSession(sessionKey);
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  closeStream(true);
});

/** 加载指定会话的线程快照并重建前端时间线。 */
async function selectSession(sessionKey: string) {
  if (!sessionKey || sessionKey === selectedSessionKey.value) {
    return;
  }

  closeStream(true);
  activeRunId.value = null;
  cancelling.value = false;
  selectedSessionKey.value = sessionKey;
  constraints.value = {};
  threadLoading.value = true;
  try {
    const thread = await fetchSessionThread(sessionKey);
    // 中文注释：用户连续点击多个历史会话时，旧请求可能比新请求更晚返回；这时直接丢掉旧结果，避免页面跳回上一条。
    if (selectedSessionKey.value !== sessionKey) {
      return;
    }
    hydrateThread(thread);
  } catch (error) {
    // 中文注释：404 通常表示左侧历史里这条会话已经被后端删掉或清理了，继续停在这里只会反复报错。
    if (error instanceof ApiRequestError && error.status === 404 && selectedSessionKey.value === sessionKey) {
      resetToBlankWorkspace();
      emit("update:selectedKey", "");
      emit("refreshSessions");
      handleError(error, "会话不存在，已回到空白工作台");
      return;
    }
    handleError(error, "加载会话线程失败");
  } finally {
    // 中文注释：只关闭当前这次选择对应的加载状态，避免旧请求影响用户后来点开的新会话。
    if (selectedSessionKey.value === sessionKey) {
      threadLoading.value = false;
    }
  }
}

/**
 * 中文注释：新窗口或新会话还没有真正执行时，中间只保留输入区。
 * 这样用户第一眼看到的是“要研究什么”，不会被空时间线卡片打断思路。
 */
function resetToBlankWorkspace() {
  closeStream(true);
  selectedSessionKey.value = "";
  selectedTitle.value = "新的研究主题";
  constraints.value = {};
  selectedRunStartedAt.value = null;
  timelineSnapshot.value = null;
  sending.value = false;
  cancelling.value = false;
  activeRunId.value = null;
  threadLoading.value = false;
}

/** 提交当前主题，必要时自动创建会话，并接入对应的 SSE 流。 */
async function submitTopic() {
  const content = draft.value.trim();
  if (!content || sending.value || isRunning.value) {
    return;
  }

  sending.value = true;
  const submittedContent = content;
  draft.value = "";

  try {
    const sessionKey = await ensureActiveSession();
    const turnId = createReadableId();
    aggregator.addOptimisticUserMessage(submittedContent, turnId);
    syncSnapshot();
    const accepted = await startSessionRun(sessionKey, {
      content: submittedContent,
      turn_id: turnId,
      constraints: { ...constraints.value },
    });
    emit("refreshSessions");
    activeRunId.value = accepted.run_id;
    openStream(sessionKey, accepted.stream_url);
  } catch (error) {
    draft.value = submittedContent;
    await reloadCurrentThread();
    handleError(error, "启动工作流失败");
    sending.value = false;
  }
}

/** 中文注释：继续执行时，前端只告诉后端“恢复当前会话”，具体恢复位置由后端从历史里查。 */
async function resumeLatestCheckpoint() {
  if (!selectedSessionKey.value || sending.value || isRunning.value) {
    return;
  }

  sending.value = true;
  const content = "继续上次失败的位置";
  try {
    const turnId = createReadableId();
    aggregator.addOptimisticUserMessage(content, turnId);
    syncSnapshot();
    const accepted = await startSessionRun(selectedSessionKey.value, {
      content,
      turn_id: turnId,
      resume_from_last_checkpoint: true,
    });
    emit("refreshSessions");
    activeRunId.value = accepted.run_id;
    openStream(selectedSessionKey.value, accepted.stream_url);
  } catch (error) {
    await reloadCurrentThread();
    handleError(error, "继续执行失败");
    sending.value = false;
  }
}

/** 为实时流建立 EventSource 订阅，并把后端发来的每条消息交给前端聚合器整理。 */
function openStream(sessionKey: string, streamUrl: string) {
  closeStream(true);
  manualClose.value = false;
  streamSource.value = subscribeSessionRun(streamUrl, {
    onOpen: () => {
      sending.value = false;
    },
    onEvent: async (event) => {
      aggregator.apply(event);
      syncSnapshot();
      if (event.run_started_at) {
        selectedRunStartedAt.value = event.run_started_at;
      }
      if (event.event === "turn_end") {
        await handleRunFinished(sessionKey, event);
      }
    },
    onError: async () => {
      if (manualClose.value) {
        return;
      }
      closeStream(true);
      sending.value = false;
      pushToast({
        tone: "error",
        title: "流式连接已中断",
        description: "正在尝试使用最新线程快照恢复页面状态。",
      });
      await reloadCurrentThread();
      emit("refreshSessions");
    },
  });
}

/** 向后端发送真正的停止请求；这里只关闭 SSE 会让后台任务继续运行。 */
async function cancelActiveRun() {
  const runId = activeRunId.value;
  if (!selectedSessionKey.value || !runId || cancelling.value) {
    return;
  }

  cancelling.value = true;
  try {
    await cancelSessionRun(selectedSessionKey.value, runId);
  } catch (error) {
    cancelling.value = false;
    handleError(error, "停止工作流失败");
  }
}

/** 当 run 结束时刷新线程和列表，确保左侧历史与中间时间线都展示落库后的结果。 */
async function handleRunFinished(sessionKey: string, event: SessionRuntimeEvent) {
  closeStream(true);
  sending.value = false;
  cancelling.value = false;
  activeRunId.value = null;
  await reloadCurrentThread();
  emit("refreshSessions");
  if (event.status === "failed") {
    pushToast({
      tone: "error",
      title: "工作流执行失败",
      description: event.message ?? event.content ?? "请查看时间线中的错误信息。",
    });
    return;
  }
  if (event.status === "cancelled") {
    pushToast({
      tone: "info",
      title: "工作流已停止",
      description: "已保留停止前已经完成的结果。",
    });
    return;
  }
  pushToast({
    tone: "success",
    title: "工作流已完成",
    description: "最新检索结果和产物清单已同步到当前会话。",
  });
  if (selectedSessionKey.value !== sessionKey) {
    emit("update:selectedKey", sessionKey);
  }
}

/** 若当前还没有活动会话，则自动创建一个空会话再继续提交流程。 */
async function ensureActiveSession() {
  if (selectedSessionKey.value) {
    return selectedSessionKey.value;
  }
  const payload = await createSession();
  hydrateThread(emptyThreadFromSummary(payload.session));
  selectedSessionKey.value = payload.session.key;
  emit("update:selectedKey", payload.session.key);
  emit("refreshSessions");
  return payload.session.key;
}

/** 关闭当前 EventSource，避免切换会话后仍然消费旧流。 */
function closeStream(markAsManual: boolean) {
  manualClose.value = markAsManual;
  if (streamSource.value) {
    streamSource.value.close();
    streamSource.value = null;
  }
}

/** 使用线程快照重建视图状态，并同步标题和时间线。 */
function hydrateThread(thread: SessionThread) {
  selectedTitle.value = thread.title || "会话时间线";
  selectedRunStartedAt.value = thread.run_started_at;
  aggregator.hydrate(thread);
  syncSnapshot();
}

/** 把聚合器当前快照写回响应式状态，驱动页面刷新。 */
function syncSnapshot() {
  timelineSnapshot.value = aggregator.snapshot();
  selectedRunStartedAt.value = timelineSnapshot.value.runStartedAt;
}

/** 重新拉取当前会话线程，用于 run 完成或断流后的状态修复。 */
async function reloadCurrentThread() {
  if (!selectedSessionKey.value) {
    return;
  }
  const thread = await fetchSessionThread(selectedSessionKey.value);
  hydrateThread(thread);
}

/** 根据会话摘要构造一个空线程，便于新建会话后立即切换 UI。 */
function emptyThreadFromSummary(summary: SessionSummary): SessionThread {
  return {
    key: summary.key,
    title: summary.title,
    status: summary.status,
    messages: [],
    events: [],
    artifacts: [],
    has_pending_tool_calls: false,
    run_started_at: summary.run_started_at,
  };
}

/**
 * 中文注释：中间时间线现在只展示“执行过程”。
 * 消息内容和产物文件虽然还会保存在会话里，但页面不再展示它们，所以这里只看真正会显示出来的执行事件。
 */
function hasTimelineContent(snapshot: SessionTimelineSnapshot | null) {
  return Boolean(snapshot?.runtimeEvents.length);
}

function statusTone(status: string) {
  if (status === "completed") {
    return "success";
  }
  if (status === "running") {
    return "warning";
  }
  if (status === "failed") {
    return "danger";
  }
  return "neutral";
}

function statusLabel(status: string) {
  if (status === "running") {
    return "运行中";
  }
  if (status === "completed") {
    return "已完成";
  }
  if (status === "failed") {
    return "失败";
  }
  return "准备中";
}

/** 统一把异常转换为 toast，减少页面分散的错误处理分支。 */
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
  <section
    class="page-shell session-workspace-shell"
    :data-timeline-visible="shouldShowTimeline"
    :data-welcome-visible="shouldShowWelcomeComposer"
  >
    <header v-if="!shouldShowWelcomeComposer" class="hero-card session-hero-card session-compact-header">
      <div class="session-hero-icon" aria-hidden="true">
        <FileSearch :size="24" />
      </div>
      <div class="hero-copy session-compact-copy">
        <span class="eyebrow">Workflow Session</span>
        <h1>{{ selectedTitle }}</h1>
        <p>
          {{ isBlankWorkspace ? "从一个研究主题开始，系统会在执行后自动展开 Live Timeline。" : "历史在左侧管理，这里专注展示当前主题的输入和执行过程。" }}
        </p>
      </div>

      <div class="session-compact-meta">
        <StatusPill :tone="statusTone(currentStatus)" :label="taskStatusLabel" />
        <span class="session-compact-date"><CalendarDays :size="14" />{{ compactUpdatedAt }}</span>
      </div>
    </header>

    <section class="session-workbench">
      <div class="session-main-column">
        <div class="session-content-area">
          <SessionComposer
            v-if="shouldShowWelcomeComposer"
            v-model="draft"
            variant="welcome"
            heading="今天想探索什么研究方向？"
            helper-text="输入论文主题、研究问题或调研任务，系统会自动创建会话并展开执行时间线。"
            placeholder="例如：多智能体论文写作系统的研究现状与关键挑战"
            :rows="3"
            :running="isRunning"
            :sending="sending || props.creatingSession"
            :cancellable="Boolean(activeRunId)"
            :cancelling="cancelling"
            :status-text="statusText"
            v-model:constraints="constraints"
            @submit="submitTopic"
            @cancel="cancelActiveRun"
          />

          <SessionTimeline
            v-else-if="shouldShowTimeline"
            :title="selectedTitle"
            :snapshot="timelineSnapshot"
            :loading="threadLoading"
            @resume="resumeLatestCheckpoint"
          />

          <section v-if="finalResultMarkdown" class="session-final-result">
            <div class="session-final-result-head"><h3>最终结果</h3></div>
            <pre class="session-final-result-body">{{ finalResultMarkdown }}</pre>
          </section>

          <div v-else-if="threadLoading" class="session-empty session-workbench-loading">
            <LoaderCircle class="spinning" :size="18" />
            <span>正在准备会话工作台…</span>
          </div>
        </div>

        <SessionComposer
          v-if="!shouldShowWelcomeComposer"
          v-model="draft"
          :running="isRunning"
          :sending="sending || props.creatingSession"
          :cancellable="Boolean(activeRunId)"
          :cancelling="cancelling"
          :status-text="statusText"
          v-model:constraints="constraints"
          @submit="submitTopic"
          @cancel="cancelActiveRun"
        />
      </div>

      <aside v-if="shouldShowTimeline" class="session-insights" aria-label="任务概览">
        <section class="insight-card insight-status-card">
          <div class="insight-card-heading"><h2>任务状态</h2><Gauge :size="16" /></div>
          <div class="insight-status-value">
            <span class="insight-status-icon"><CheckCircle2 :size="24" /></span>
            <strong>{{ taskStatusLabel }}</strong>
          </div>
          <p>{{ currentStatus === "completed" ? "任务已成功完成所有步骤" : currentStatus === "cancelled" ? "已保留停止前的处理进度" : "任务正在按照流程执行" }}</p>
        </section>

        <section class="insight-card">
          <div class="insight-card-heading"><h2>用时统计</h2><Timer :size="16" /></div>
          <div class="insight-metric-grid">
            <div><strong>{{ timelineStats.duration }}</strong><span>总耗时</span></div>
            <div><strong>{{ timelineStats.completed }}</strong><span>完成步骤</span></div>
            <div><strong>{{ timelineStats.total ? Math.round((timelineStats.completed / timelineStats.total) * 100) : 0 }}%</strong><span>完成度</span></div>
            <div><strong>{{ timelineStats.total }}</strong><span>执行步骤</span></div>
          </div>
        </section>

        <section class="insight-card">
          <div class="insight-card-heading"><h2>资源使用</h2><Upload :size="16" /></div>
          <div class="insight-token-total">{{ formatNumber(timelineStats.inputTokens + timelineStats.outputTokens) }} <span>Token 使用量</span></div>
          <div class="insight-progress"><span :style="{ width: `${Math.min(100, timelineStats.total ? 45 + timelineStats.completed / timelineStats.total * 55 : 0)}%` }"></span></div>
          <div class="insight-token-breakdown">
            <div><span>输入 Token</span><strong>{{ formatNumber(timelineStats.inputTokens) }}</strong></div>
            <div><span>输出 Token</span><strong>{{ formatNumber(timelineStats.outputTokens) }}</strong></div>
          </div>
        </section>

        <section class="insight-card insight-output-card">
          <div class="insight-card-heading"><h2>输出结果</h2><PackageOpen :size="16" /></div>
          <template v-if="finalArtifacts.length">
            <a
              v-for="artifact in finalArtifacts"
              :key="artifact.id"
              :href="artifactUrlFor(artifact.id)"
              target="_blank"
              rel="noreferrer"
              class="insight-artifact"
            >
              <span class="insight-artifact-icon"><FileDown :size="16" /></span>
              <span><strong>{{ artifact.name }}</strong><small>{{ formatArtifactSize(artifact.size) }} · 点击打开</small></span>
            </a>
          </template>
          <div v-else class="insight-no-output"><PackageOpen :size="18" /><span>任务完成后会显示最终论文</span></div>
        </section>
      </aside>
    </section>
  </section>
</template>
