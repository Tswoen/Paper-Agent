import { createRouter, createWebHistory } from "vue-router";

import SystemSettingsView from "../views/SystemSettingsView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      redirect: "/settings",
    },
    {
      path: "/settings",
      name: "settings",
      component: SystemSettingsView,
    },
  ],
});

export default router;
