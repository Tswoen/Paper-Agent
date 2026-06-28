import { reactive } from "vue";

export type ToastTone = "success" | "error" | "info";

export interface ToastMessage {
  id: number;
  title: string;
  description?: string;
  tone: ToastTone;
}

export const notifications = reactive({
  items: [] as ToastMessage[],
});

let nextId = 1;

export function pushToast(message: Omit<ToastMessage, "id">) {
  const item: ToastMessage = {
    id: nextId++,
    ...message,
  };
  notifications.items.push(item);
  window.setTimeout(() => {
    removeToast(item.id);
  }, 3200);
}

export function removeToast(id: number) {
  const index = notifications.items.findIndex((item) => item.id === id);
  if (index >= 0) {
    notifications.items.splice(index, 1);
  }
}
