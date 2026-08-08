let fallbackCounter = 0;
const READABLE_ID_RANDOM_LENGTH = 12;
const READABLE_ID_CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789";

/**
 * 生成“年月日时分秒_随机字符串”格式的编号。
 *
 * 回合编号会写入会话记录和产物目录，因此把创建时间放在开头，查看编号时就能直接知道大概的创建时间。
 * 同一秒可能连续提交多次，后面的 12 位随机字符用来区分这些提交。
 */
export function createReadableId() {
  const now = new Date();
  const timestamp = [
    now.getFullYear(),
    now.getMonth() + 1,
    now.getDate(),
    now.getHours(),
    now.getMinutes(),
    now.getSeconds(),
  ]
    .map((part, index) => (index === 0 ? String(part) : String(part).padStart(2, "0")))
    .join("");

  return `${timestamp}_${createReadableRandomPart()}`;
}

/** 生成编号末尾的随机字符；优先使用浏览器提供的随机数据。 */
function createReadableRandomPart() {
  const browserCrypto = globalThis.crypto;
  if (browserCrypto?.getRandomValues) {
    let randomPart = "";

    // 只取 0 到 251 的数字，这样每个字符被选中的机会完全相同。
    while (randomPart.length < READABLE_ID_RANDOM_LENGTH) {
      const bytes = new Uint8Array(READABLE_ID_RANDOM_LENGTH);
      browserCrypto.getRandomValues(bytes);
      for (const byte of bytes) {
        if (byte < 252) {
          randomPart += READABLE_ID_CHARACTERS[byte % READABLE_ID_CHARACTERS.length];
        }
        if (randomPart.length === READABLE_ID_RANDOM_LENGTH) {
          return randomPart;
        }
      }
    }
  }

  // 少数旧环境没有浏览器随机功能时，仍然生成固定长度的编号，避免影响正常提交。
  fallbackCounter += 1;
  let randomPart = `${Date.now().toString(36)}${fallbackCounter.toString(36)}`;
  while (randomPart.length < READABLE_ID_RANDOM_LENGTH) {
    randomPart += Math.random().toString(36).slice(2);
  }
  return randomPart.slice(-READABLE_ID_RANDOM_LENGTH);
}

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
