const DEFAULT_MAX_ECHO = 80;

const patterns = {
  greeting: [/^(hi|hello|hey)\b/i, /^(你好|您好|嗨|哈喽)/],
  thanks: [/\b(thanks|thank you)\b/i, /谢谢|多谢/],
  farewell: [/\b(bye|goodbye|see you)\b/i, /再见|拜拜/],
  help: [/\b(help|support)\b/i, /能做什么|可以做什么|帮助/],
  whoami: [/你是谁|你的名字|叫什么/]
};

const responses = {
  greeting: '你好！很高兴和你交流。',
  thanks: '不客气，如果还有需要，随时告诉我。',
  farewell: '再见！祝你今天愉快。',
  help: '我可以进行日常交流，并帮助你处理文件或图片信息，例如读取文件内容、判断是否为图片、汇总文件大小等。',
  whoami: '我是一个可以帮你聊天并处理文件的小助手。'
};

function truncate(text, maxLength) {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength)}...`;
}

function respondToMessage(message, options = {}) {
  const maxEchoLength = Number.isInteger(options.maxEchoLength)
    ? options.maxEchoLength
    : DEFAULT_MAX_ECHO;

  if (typeof message !== 'string') {
    return '我还需要你提供具体的文字内容，才能继续交流。';
  }

  const trimmed = message.trim();
  if (!trimmed) {
    return '我在这儿，告诉我你想聊什么吧。';
  }

  for (const [intent, intentPatterns] of Object.entries(patterns)) {
    if (intentPatterns.some((pattern) => pattern.test(trimmed))) {
      return responses[intent];
    }
  }

  const echo = truncate(trimmed, maxEchoLength);
  return `我明白了：${echo}。如果需要文件或图片方面的帮助，也可以告诉我。`;
}

module.exports = {
  respondToMessage
};
