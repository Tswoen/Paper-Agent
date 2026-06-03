<template>
  <main class="main-content" :class="{ 'sidebar-collapsed': collapsed }">
    <section class="content-wrapper">
      <div v-if="isLoading" class="loading-overlay">
        <div class="loading-spinner"></div>
        <p>正在加载...</p>
      </div>

      <router-view v-slot="{ Component }">
        <transition name="page-fade" mode="out-in">
          <keep-alive>
            <component :is="Component" @loading="handleLoading" />
          </keep-alive>
        </transition>
      </router-view>
    </section>
  </main>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'

defineProps({
  collapsed: {
    type: Boolean,
    default: false
  }
})

const route = useRoute()
const isLoading = ref(false)

const handleLoading = (loading) => {
  isLoading.value = loading
}

watch(() => route.path, () => {
  isLoading.value = false
})
</script>

<style scoped>
.main-content {
  flex: 1;
  min-width: 0;
  height: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  background: var(--pa-bg);
}

.content-wrapper {
  position: relative;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.loading-overlay {
  position: absolute;
  inset: 0;
  z-index: 100;
  display: grid;
  place-items: center;
  align-content: center;
  gap: 12px;
  background: color-mix(in srgb, var(--pa-surface) 88%, transparent);
  color: var(--pa-text-muted);
}

.loading-spinner {
  width: 38px;
  height: 38px;
  border: 4px solid var(--pa-border);
  border-top-color: var(--pa-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.loading-overlay p {
  margin: 0;
  font-size: 0.9rem;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.page-fade-enter-active,
.page-fade-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.page-fade-enter-from {
  opacity: 0;
  transform: translateY(8px);
}

.page-fade-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
