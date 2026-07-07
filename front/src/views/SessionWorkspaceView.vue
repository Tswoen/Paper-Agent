<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

import {
  createSession,
  deleteSession,
  fetchSessionThread,
  listSessions,
  startSessionRun,
  subscribeSessionRun,
} from "../api/sessions";
import SessionComposer from "../components/session/SessionComposer.vue";
import SessionSidebar from "../components/session/SessionSidebar.vue";
import SessionTimeline from "../components/session/SessionTimeline.vue";
import { SessionStreamAggregator } from "../lib/session-stream-aggregator";
import { pushToast } from "../stores/notifications";
import type {
  SessionRuntimeEvent,
  SessionSummary,
  SessionThread,
  SessionTimelineSnapshot,
} from "../types/sessions";

const sessions = ref<SessionSummary[]>([]);
const selectedSessionKey = ref("");
const selectedTitle = ref("会话时间线");
const draft = ref("");
const listLoading = ref(true);
const threadLoading = ref(false);
const creating = ref(false);
const sending = ref(false);
const streamSource = ref<EventSource | null>(null);
const manualClose = ref(false);
const timelineSnapshot = ref<SessionTimelineSnapshot | null>(null);

const aggregator = new SessionStreamAggregator();

const isRunning = computed(() => timelineSnapshot.value?.isStreaming ?? false);
const statusText = computed(() => {
  if (sending.value) {
    return "正在创建运行并建立流式连接";
  }
  if (isRunning.value) {
    return "工作流正在执行中，当前会话不可重复提交";
  }
  return "输入主题后即可启动一次新的论文工作流";
});

onMounted(async () => {
  await initializeWorkspace();
});

onBeforeUnmount(() => {
  closeStream(true);
});

/** 页面首次进入时加载会话列表，并尽量打开最近一条会话。 */
async function initializeWorkspace() {
  listLoading.value = true;
  try {
    const payload = await listSessions();
    sessions.value = payload.sessions;
    if (payload.sessions.length > 0) {
      await selectSession(payload.sessions[0].key);
    } else {
      timelineSnapshot.value = null;
      selectedTitle.value = "会话时间线";
    }
  } catch (error) {
    handleError(error, "加载会话列表失败");
  } finally {
    listLoading.value = false;
  }
}

/** 从后端刷新会话列表，并尽量保持当前选中的会话不变。 */
async function refreshSessions() {
  const payload = await listSessions();
  sessions.value = payload.sessions;
  if (!selectedSessionKey.value && payload.sessions.length > 0) {
    selectedSessionKey.value = payload.sessions[0].key;
  }
}

/** 加载指定会话的线程快照并重建前端时间线。 */
async function selectSession(sessionKey: string) {
  if (!sessionKey) {
    return;
  }
  closeStream(true);
  selectedSessionKey.value = sessionKey;
  threadLoading.value = true;
  try {
    const thread = await fetchSessionThread(sessionKey);
    hydrateThread(thread);
  } catch (error) {
    handleError(error, "加载会话线程失败");
  } finally {
    threadLoading.value = false;
  }
}

/** 新建空会话，并立即切换到该会话工作台。 */
async function createNewSession() {
  creating.value = true;
  try {
    const payload = await createSession();
    sessions.value = [payload.session, ...sessions.value];
    hydrateThread(emptyThreadFromSummary(payload.session));
    selectedSessionKey.value = payload.session.key;
  } catch (error) {
    handleError(error, "创建会话失败");
  } finally {
    creating.value = false;
  }
}

/** 删除指定会话，并在必要时把焦点切换到下一条可用会话。 */
async function removeSession(sessionKey: string) {
  try {
    await deleteSession(sessionKey);
    const wasSelected = selectedSessionKey.value === sessionKey;
    sessions.value = sessions.value.filter((item) => item.key !== sessionKey);
    if (wasSelected) {
      closeStream(true);
      if (sessions.value.length > 0) {
        await selectSession(sessions.value[0].key);
      } else {
        selectedSessionKey.value = "";
        selectedTitle.value = "会话时间线";
        timelineSnapshot.value = null;
      }
    }
    pushToast({
      tone: "success",
      title: "会话已删除",
      description: "该工作台和对应历史记录已从列表中移除。",
    });
  } catch (error) {
    handleError(error, "删除会话失败");
  }
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
    const turnId = crypto.randomUUID();
    aggregator.addOptimisticUserMessage(submittedContent, turnId);
    syncSnapshot();
    const accepted = await startSessionRun(sessionKey, {
      content: submittedContent,
      turn_id: turnId,
    });
    await refreshSessions();
    openStream(sessionKey, accepted.stream_url);
  } catch (error) {
    draft.value = submittedContent;
    await reloadCurrentThread();
    handleError(error, "启动工作流失败");
    sending.value = false;
  }
}

/** 为实时流建立 EventSource 订阅，并把事件持续喂给前端聚合器。 */
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
      await refreshSessions();
    },
  });
}

/** 当 run 结束时刷新线程和列表，确保历史快照与落库结果完全对齐。 */
async function handleRunFinished(sessionKey: string, event: SessionRuntimeEvent) {
  closeStream(true);
  sending.value = false;
  await reloadCurrentThread();
  await refreshSessions();
  if (event.status === "failed") {
    pushToast({
      tone: "error",
      title: "工作流执行失败",
      description: event.message ?? event.content ?? "请查看时间线中的错误信息。",
    });
    return;
  }
  pushToast({
    tone: "success",
    title: "工作流已完成",
    description: "最新检索结果和产物清单已同步到当前会话。",
  });
  if (selectedSessionKey.value !== sessionKey) {
    await selectSession(sessionKey);
  }
}

/** 若当前还没有活动会话，则自动创建一个空会话再继续提交流程。 */
async function ensureActiveSession() {
  if (selectedSessionKey.value) {
    return selectedSessionKey.value;
  }
  const payload = await createSession();
  sessions.value = [payload.session, ...sessions.value];
  hydrateThread(emptyThreadFromSummary(payload.session));
  selectedSessionKey.value = payload.session.key;
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
  aggregator.hydrate(thread);
  syncSnapshot();
}

/** 把聚合器当前快照写回响应式状态，驱动页面刷新。 */
function syncSnapshot() {
  timelineSnapshot.value = aggregator.snapshot();
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
  <section class="page-shell session-workspace-shell">
    <header class="hero-card session-hero-card">
      <div class="hero-copy">
        <span class="eyebrow">Workflow Session</span>
        <h1>主题驱动的论文工作流会话台</h1>
        <p>
          左侧管理历史会话，中间按时间线回放实时输出，底部输入论文主题后即可
          触发检索流程并持续看到 reasoning、进度和产物信息。
        </p>
      </div>
    </header>

    <section class="session-workspace-grid">
      <SessionSidebar
        :sessions="sessions"
        :selected-key="selectedSessionKey"
        :loading="listLoading"
        :creating="creating"
        @create="createNewSession"
        @select="selectSession"
        @remove="removeSession"
      />

      <div class="session-main-column">
        <SessionTimeline
          :title="selectedTitle"
          :snapshot="timelineSnapshot"
          :loading="threadLoading"
        />

        <SessionComposer
          v-model="draft"
          :running="isRunning"
          :sending="sending"
          :status-text="statusText"
          @submit="submitTopic"
        />
      </div>
    </section>
  </section>
</template>
