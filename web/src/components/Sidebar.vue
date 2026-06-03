<template>
  <aside class="sidebar" :class="{ collapsed }">
    <header class="sidebar-header">
      <button class="brand-button" type="button" title="报告生成" @click="navigate('/')">
        <span class="brand-mark">PA</span>
        <span v-if="!collapsed" class="brand-copy">
          <strong>Paper Agent</strong>
          <small>Academic Research</small>
        </span>
      </button>

      <button
        class="collapse-button"
        type="button"
        :title="collapsed ? '展开导航' : '收起导航'"
        @click="$emit('toggle')"
      >
        {{ collapsed ? '›' : '‹' }}
      </button>
    </header>

    <nav class="nav-list" aria-label="主导航">
      <button
        v-for="item in navItems"
        :key="item.path"
        type="button"
        class="nav-item"
        :class="{ active: isActive(item.path) }"
        :title="collapsed ? item.title : ''"
        @click="navigate(item.path)"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span v-if="!collapsed" class="nav-text">{{ item.title }}</span>
      </button>
    </nav>

    <footer class="sidebar-footer">
      <button
        class="theme-toggle"
        type="button"
        :title="theme === 'light' ? '切换到 Dark' : '切换到 Light'"
        @click="$emit('toggle-theme')"
      >
        <span class="theme-dot">{{ theme === 'light' ? 'L' : 'D' }}</span>
        <span v-if="!collapsed" class="theme-label">
          <strong>{{ theme === 'light' ? 'Light' : 'Dark' }}</strong>
          <small>全局主题</small>
        </span>
      </button>
    </footer>
  </aside>
</template>

<script setup>
import { ref } from 'vue'
import { useRoute } from 'vue-router'

defineProps({
  collapsed: {
    type: Boolean,
    default: false
  },
  theme: {
    type: String,
    default: 'light'
  }
})

const emit = defineEmits(['toggle', 'navigate', 'toggle-theme'])
const route = useRoute()

const navItems = ref([
  { title: '报告生成', path: '/', icon: 'R' },
  { title: '系统配置', path: '/configuration', icon: 'C' }
])

const isActive = (path) => {
  return path === '/' ? route.path === '/' : route.path.startsWith(path)
}

const navigate = (path) => {
  emit('navigate', path)
}
</script>

<style scoped>
.sidebar {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  width: 236px;
  min-width: 236px;
  height: 100%;
  background: var(--pa-surface);
  border-right: 1px solid var(--pa-border);
  transition: width 0.22s ease, min-width 0.22s ease;
}

.sidebar.collapsed {
  width: 68px;
  min-width: 68px;
}

.sidebar-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  align-items: center;
  min-height: 68px;
  padding: 12px;
  border-bottom: 1px solid var(--pa-border);
}

.sidebar.collapsed .sidebar-header {
  grid-template-columns: 1fr;
  justify-items: center;
  gap: 10px;
  padding: 10px 8px;
}

.brand-button {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  border: 0;
  background: transparent;
  color: var(--pa-text);
  text-align: left;
  cursor: pointer;
}

.brand-mark,
.nav-icon,
.theme-dot {
  display: grid;
  place-items: center;
  flex: 0 0 auto;
  border-radius: 8px;
  font-weight: 800;
}

.brand-mark {
  width: 38px;
  height: 38px;
  background: var(--pa-primary-soft);
  color: var(--pa-primary);
  letter-spacing: 0;
}

.brand-copy {
  display: grid;
  gap: 2px;
  min-width: 0;
}

.brand-copy strong,
.brand-copy small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.brand-copy strong {
  color: var(--pa-text);
  font-size: 0.98rem;
}

.brand-copy small {
  color: var(--pa-text-subtle);
  font-size: 0.74rem;
}

.collapse-button {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-surface);
  color: var(--pa-text-muted);
  font-size: 1.25rem;
  cursor: pointer;
}

.collapse-button:hover {
  background: var(--pa-surface-soft);
}

.nav-list {
  display: grid;
  align-content: start;
  gap: 8px;
  min-height: 0;
  padding: 14px 10px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: 42px;
  padding: 0 10px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: var(--pa-text-muted);
  text-align: left;
  cursor: pointer;
}

.sidebar.collapsed .nav-item {
  justify-content: center;
  padding: 0;
}

.nav-item:hover {
  background: var(--pa-surface-soft);
  color: var(--pa-text);
}

.nav-item.active {
  border-color: color-mix(in srgb, var(--pa-primary) 35%, transparent);
  background: var(--pa-primary-soft);
  color: var(--pa-primary-strong);
}

.nav-icon {
  width: 26px;
  height: 26px;
  background: var(--pa-surface-muted);
  color: currentColor;
  font-size: 0.78rem;
}

.nav-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 700;
}

.sidebar-footer {
  padding: 12px 10px;
  border-top: 1px solid var(--pa-border);
}

.theme-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: 44px;
  padding: 0 10px;
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  background: var(--pa-surface-soft);
  color: var(--pa-text);
  text-align: left;
  cursor: pointer;
}

.sidebar.collapsed .theme-toggle {
  justify-content: center;
  padding: 0;
}

.theme-toggle:hover {
  border-color: var(--pa-border-strong);
}

.theme-dot {
  width: 26px;
  height: 26px;
  background: var(--pa-surface);
  color: var(--pa-primary);
  font-size: 0.76rem;
}

.theme-label {
  display: grid;
  gap: 2px;
  min-width: 0;
}

.theme-label strong,
.theme-label small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.theme-label strong {
  font-size: 0.86rem;
}

.theme-label small {
  color: var(--pa-text-subtle);
  font-size: 0.72rem;
}
</style>
