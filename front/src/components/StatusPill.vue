<script setup lang="ts">
defineProps<{
  tone?: "neutral" | "success" | "warning" | "danger";
  label: string;
}>();
</script>

<template>
  <span class="status-pill" :data-tone="tone ?? 'neutral'">
    {{ label }}
  </span>
</template>
