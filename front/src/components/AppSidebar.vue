<script setup lang="ts">
import { ChevronLeft, PanelsTopLeft, Workflow } from "lucide-vue-next";
import { RouterLink, useRoute } from "vue-router";

defineProps<{
  collapsed: boolean;
}>();

const emit = defineEmits<{
  toggle: [];
}>();

const route = useRoute();
</script>

<template>
  <aside class="sidebar" :data-collapsed="collapsed">
    <div class="sidebar-head">
      <div class="brand-lockup">
        <div class="brand-mark">PA</div>
        <div v-if="!collapsed" class="brand-copy">
          <span class="eyebrow">Papers Agents</span>
          <strong>Console</strong>
        </div>
      </div>
      <button
        class="icon-button"
        type="button"
        :aria-label="collapsed ? '展开导航' : '收起导航'"
        @click="emit('toggle')"
      >
        <ChevronLeft :size="16" :class="{ rotated: collapsed }" />
      </button>
    </div>

    <nav class="sidebar-nav" aria-label="主导航">
      <RouterLink
        class="nav-item"
        :class="{ active: route.name === 'sessions' }"
        to="/sessions"
      >
        <Workflow :size="18" />
        <span v-if="!collapsed">会话工作台</span>
      </RouterLink>

      <RouterLink
        class="nav-item"
        :class="{ active: route.name === 'settings' }"
        to="/settings"
      >
        <PanelsTopLeft :size="18" />
        <span v-if="!collapsed">系统配置</span>
      </RouterLink>
    </nav>

    <div v-if="!collapsed" class="sidebar-foot">
      <p>工作流会话台支持按主题发起检索，并通过流式时间线展示运行过程。</p>
    </div>
  </aside>
</template>
