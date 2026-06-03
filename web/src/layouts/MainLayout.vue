<template>
  <div class="main-layout" :data-theme="theme">
    <Sidebar
      :collapsed="sidebarCollapsed"
      :theme="theme"
      @toggle="toggleSidebar"
      @navigate="handleNavigate"
      @toggle-theme="toggleTheme"
    />
    <MainContent :collapsed="sidebarCollapsed" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import Sidebar from '../components/Sidebar.vue'
import MainContent from '../components/MainContent.vue'

const router = useRouter()
const sidebarCollapsed = ref(false)
const theme = ref('light')

const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

const toggleTheme = () => {
  theme.value = theme.value === 'light' ? 'dark' : 'light'
  localStorage.setItem('paper-agent-theme', theme.value)
}

const handleNavigate = (path) => {
  router.push(path)
}

onMounted(() => {
  const savedTheme = localStorage.getItem('paper-agent-theme')
  if (savedTheme === 'light' || savedTheme === 'dark') {
    theme.value = savedTheme
  }
})
</script>

<style scoped>
.main-layout {
  display: flex;
  width: 100%;
  height: 100vh;
  overflow: hidden;
  background: var(--pa-bg);
  color: var(--pa-text);
}
</style>
