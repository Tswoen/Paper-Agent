let fallbackCounter = 0;

/**
 * 中文注释：
 * 这个方法专门负责生成前端要用的临时 ID。
 * 之前页面直接调用 crypto.randomUUID()，
 * 一旦当前运行环境没有这个方法，就会马上报错。
 *
 * 这里做了三层兜底：
 * 1. 能直接用 randomUUID，就直接用。
 * 2. 没有 randomUUID，但有 getRandomValues，就自己拼一个 UUID 风格的字符串。
 * 3. 如果上面两个都没有，就退回到“时间 + 计数器 + 随机数”的简单方案。
 *
 * 这样就算页面跑在兼容性一般的环境里，也不会因为生成 ID 失败把整个交流程打断。
 */
export function createRandomId(prefix = "id") {
  const browserCrypto = globalThis.crypto;

  if (browserCrypto?.randomUUID) {
    return browserCrypto.randomUUID();
  }

  if (browserCrypto?.getRandomValues) {
    const bytes = new Uint8Array(16);
    browserCrypto.getRandomValues(bytes);

    // 中文注释：
    // 这里把随机字节整理成常见的 UUID v4 样式，
    // 方便统一展示，也更方便我们以后排查问题。
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    const hex = Array.from(bytes, (item) => item.toString(16).padStart(2, "0"));
    return `${hex[0]}${hex[1]}${hex[2]}${hex[3]}-${hex[4]}${hex[5]}-${hex[6]}${hex[7]}-${hex[8]}${hex[9]}-${hex[10]}${hex[11]}${hex[12]}${hex[13]}${hex[14]}${hex[15]}`;
  }

  fallbackCounter += 1;
  return `${prefix}-${Date.now()}-${fallbackCounter}-${Math.random().toString(16).slice(2)}`;
}
