<template>
  <div id="app">
    <router-view />
  </div>
</template>

<script setup>
</script>

<style>
:root {
  --pa-bg: #f6f7f8;
  --pa-surface: #ffffff;
  --pa-surface-soft: #f9fafb;
  --pa-surface-muted: #f1f4f5;
  --pa-border: #e2e7e9;
  --pa-border-strong: #cbd5da;
  --pa-text: #20272c;
  --pa-text-muted: #69757d;
  --pa-text-subtle: #8a949a;
  --pa-primary: #2f8da3;
  --pa-primary-soft: #e8f5f8;
  --pa-primary-strong: #1f6f82;
  --pa-warning: #a86a1f;
  --pa-warning-soft: #fff6e8;
  --pa-danger: #b44f55;
  --pa-danger-soft: #fff0f1;
  --pa-success: #287d65;
  --pa-success-soft: #eaf7f1;
  --pa-shadow: 0 10px 30px rgba(24, 32, 38, 0.07);
  --pa-input-bg: #ffffff;
  --pa-scrollbar: #cfd8dc;
}

[data-theme='dark'] {
  --pa-bg: #11161a;
  --pa-surface: #171d22;
  --pa-surface-soft: #1d242a;
  --pa-surface-muted: #232b32;
  --pa-border: #2d373f;
  --pa-border-strong: #3d4a54;
  --pa-text: #e6ebee;
  --pa-text-muted: #a7b2ba;
  --pa-text-subtle: #7f8c95;
  --pa-primary: #7fc7d8;
  --pa-primary-soft: #17323a;
  --pa-primary-strong: #9ed9e6;
  --pa-warning: #e5b46d;
  --pa-warning-soft: #332719;
  --pa-danger: #ea8d94;
  --pa-danger-soft: #351f23;
  --pa-success: #86d7b8;
  --pa-success-soft: #183329;
  --pa-shadow: 0 12px 34px rgba(0, 0, 0, 0.28);
  --pa-input-bg: #12181d;
  --pa-scrollbar: #40505a;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html,
body,
#app {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

body {
  font-family: Inter, 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
  background: #f6f7f8;
  color: #20272c;
  text-rendering: geometricPrecision;
}

button,
textarea,
input,
select {
  font: inherit;
}

button {
  letter-spacing: 0;
}
</style>
