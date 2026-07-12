<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { RouterView, useRoute, useRouter } from "vue-router";

import { createSession, deleteSession, listSessions } from "./api/sessions";
import AppSidebar from "./components/AppSidebar.vue";
import ToastStack from "./components/ToastStack.vue";
import { pushToast } from "./stores/notifications";
import type { SessionSummary } from "./types/sessions";

const route = useRoute();
const router = useRouter();

const sidebarCollapsed = ref(localStorage.getItem("pa.sidebar.collapsed") === "1");
const sessions = ref<SessionSummary[]>([]);
const sessionsLoading = ref(false);
const creatingSession = ref(false);
const selectedSessionKey = ref("");

const workspaceBindings = computed(() => ({
  sessions: sessions.value,
  selectedKey: selectedSessionKey.value,
  creatingSession: creatingSession.value,
  "onUpdate:selectedKey": selectSession,
  onRefreshSessions: handleRefreshSessions,
}));

watch(sidebarCollapsed, (value) => {
  localStorage.setItem("pa.sidebar.collapsed", value ? "1" : "0");
});

onMounted(async () => {
  await refreshSessions(true);
});

/**
 * 中文注释：从后端重新读取历史会话列表。
 * 第一次进入页面时会自动选中最新会话；后续刷新时尽量保持用户当前正在看的会话不变。
 */
async function refreshSessions(shouldPickFirst = false) {
  sessionsLoading.value = true;
  try {
    const payload = await listSessions();
    sessions.value = payload.sessions;

    const stillExists = payload.sessions.some((session) => session.key === selectedSessionKey.value);
    if (!stillExists) {
      selectedSessionKey.value = shouldPickFirst && payload.sessions.length > 0 ? payload.sessions[0].key : "";
    } else if (!selectedSessionKey.value && shouldPickFirst && payload.sessions.length > 0) {
      selectedSessionKey.value = payload.sessions[0].key;
    }
  } catch (error) {
    handleError(error, "加载历史会话失败");
  } finally {
    sessionsLoading.value = false;
  }
}

/** 中文注释：点击左侧栏的新建按钮时，先创建一条空会话，再让中间工作台切到这条会话。 */
async function createNewSession() {
  if (creatingSession.value) {
    return;
  }

  creatingSession.value = true;
  try {
    const payload = await createSession();
    sessions.value = [payload.session, ...sessions.value.filter((session) => session.key !== payload.session.key)];
    selectedSessionKey.value = payload.session.key;
    await ensureSessionsRoute();
  } catch (error) {
    handleError(error, "创建会话失败");
  } finally {
    creatingSession.value = false;
  }
}

/** 中文注释：用户在左侧历史里点某条记录时，记录 key，并确保右侧显示会话工作台页面。 */
async function selectSession(sessionKey: string) {
  if (!sessionKey) {
    selectedSessionKey.value = "";
    return;
  }

  selectedSessionKey.value = sessionKey;
  await ensureSessionsRoute();
}

/** 中文注释：删除历史会话后，如果删掉的是当前正在看的会话，就自动切到下一条，或者进入空白输入区。 */
async function removeSession(sessionKey: string) {
  const removedIndex = sessions.value.findIndex((session) => session.key === sessionKey);
  const wasSelected = selectedSessionKey.value === sessionKey;

  try {
    await deleteSession(sessionKey);
    const nextSessions = sessions.value.filter((session) => session.key !== sessionKey);
    sessions.value = nextSessions;

    if (wasSelected) {
      const fallback = nextSessions[removedIndex] ?? nextSessions[removedIndex - 1] ?? nextSessions[0];
      selectedSessionKey.value = fallback?.key ?? "";
    }

    pushToast({
      tone: "success",
      title: "会话已删除",
      description: "左侧历史记录已同步更新。",
    });
  } catch (error) {
    handleError(error, "删除会话失败");
  }
}

/** 中文注释：工作台执行开始或结束后会通知这里刷新，让左侧历史里的状态和摘要保持最新。 */
async function handleRefreshSessions() {
  await refreshSessions(false);
}

/** 中文注释：历史会话在任何页面都能点；如果当前在设置页，就先跳回会话页。 */
async function ensureSessionsRoute() {
  if (route.name !== "sessions") {
    await router.push({ name: "sessions" });
  }
}

/** 中文注释：把接口错误统一显示成右上角提示，避免页面静默失败。 */
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
  <div class="app-shell" :data-collapsed="sidebarCollapsed">
    <AppSidebar
      :collapsed="sidebarCollapsed"
      :sessions="sessions"
      :selected-key="selectedSessionKey"
      :loading-sessions="sessionsLoading"
      :creating-session="creatingSession"
      @toggle="sidebarCollapsed = !sidebarCollapsed"
      @create-session="createNewSession"
      @select-session="selectSession"
      @remove-session="removeSession"
    />
    <main class="app-main">
      <RouterView v-slot="{ Component }">
        <component
          :is="Component"
          v-bind="route.name === 'sessions' ? workspaceBindings : {}"
        />
      </RouterView>
    </main>
    <ToastStack />
  </div>
</template>
