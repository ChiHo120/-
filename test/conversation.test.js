const test = require('node:test');
const assert = require('node:assert/strict');

const { respondToMessage } = require('../src/conversation');

test('responds to greetings', () => {
  assert.ok(respondToMessage('你好').includes('你好'));
  assert.ok(respondToMessage('hello').includes('你好'));
});

test('responds to help requests', () => {
  const response = respondToMessage('可以做什么');
  assert.match(response, /文件|图片/);
});

test('responds to empty input', () => {
  assert.match(respondToMessage('   '), /告诉我/);
});

test('echoes general conversation with truncation', () => {
  const message = '这是一个非常非常长的句子，用来测试是否会被截断。'.repeat(5);
  const response = respondToMessage(message, { maxEchoLength: 20 });
  assert.match(response, /我明白了/);
  assert.ok(response.includes('...'));
});
