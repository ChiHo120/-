# Basic Conversation & File Helper

此项目提供基础的对话回复能力，并支持对文件/图片进行简单处理（例如读取文件内容、判断是否为图片、汇总文件大小等）。

## 功能

- 普通对话：对问候、感谢、告别、询问功能等进行回应
- 文件处理：读取文本文件、读取二进制文件、汇总文件信息
- 图片处理：通过扩展名判断是否为常见图片格式

## 使用示例

```js
const {
  respondToMessage,
  summarizeFile,
  readFileAsText,
  isImageFile
} = require('./src');

console.log(respondToMessage('你好'));
console.log(summarizeFile('./example.txt'));
console.log(readFileAsText('./example.txt'));
console.log(isImageFile('./photo.png'));
```

## 测试

```bash
npm test
```
