<script setup lang="ts">
import { ref, watch } from "vue";
import { RouterView } from "vue-router";

import AppSidebar from "./components/AppSidebar.vue";
import ToastStack from "./components/ToastStack.vue";

const sidebarCollapsed = ref(localStorage.getItem("pa.sidebar.collapsed") === "1");

watch(sidebarCollapsed, (value) => {
  localStorage.setItem("pa.sidebar.collapsed", value ? "1" : "0");
});
</script>

<template>
  <div class="app-shell" :data-collapsed="sidebarCollapsed">
    <AppSidebar :collapsed="sidebarCollapsed" @toggle="sidebarCollapsed = !sidebarCollapsed" />
    <main class="app-main">
      <RouterView />
    </main>
    <ToastStack />
  </div>
</template>
