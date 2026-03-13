const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
  isImageFile,
  summarizeFile,
  readFileAsText,
  readFileAsBuffer
} = require('../src/fileUtils');

function createTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'file-utils-'));
}

test('summarize and read file content', () => {
  const tempDir = createTempDir();
  try {
    const textPath = path.join(tempDir, 'note.txt');
    fs.writeFileSync(textPath, 'hello world');

    const summary = summarizeFile(textPath);
    assert.equal(summary.extension, '.txt');
    assert.equal(summary.sizeBytes, 11);
    assert.equal(summary.isImage, false);

    assert.equal(readFileAsText(textPath), 'hello world');
    assert.ok(readFileAsBuffer(textPath).length > 0);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('detects image file by extension', () => {
  const tempDir = createTempDir();
  try {
    const imagePath = path.join(tempDir, 'pixel.png');
    fs.writeFileSync(imagePath, Buffer.from([0x89, 0x50, 0x4e, 0x47]));

    assert.equal(isImageFile(imagePath), true);
    assert.equal(summarizeFile(imagePath).isImage, true);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('throws for missing file', () => {
  assert.throws(() => summarizeFile('/path/does/not/exist.txt'), /File not found/);
});
