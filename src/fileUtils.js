const fs = require('fs');
const path = require('path');

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff']);

function assertFilePath(filePath) {
  if (typeof filePath !== 'string' || filePath.trim() === '') {
    throw new TypeError('filePath must be a non-empty string');
  }
}

function resolveFilePath(filePath) {
  assertFilePath(filePath);
  return path.resolve(filePath);
}

function isImageFile(filePath) {
  const resolvedPath = resolveFilePath(filePath);
  return IMAGE_EXTENSIONS.has(path.extname(resolvedPath).toLowerCase());
}

function summarizeFile(filePath) {
  const resolvedPath = resolveFilePath(filePath);

  let stats;
  try {
    stats = fs.statSync(resolvedPath);
  } catch (error) {
    throw new Error(`File not found: ${resolvedPath}`);
  }

  if (!stats.isFile()) {
    throw new Error(`Path is not a file: ${resolvedPath}`);
  }

  return {
    path: resolvedPath,
    name: path.basename(resolvedPath),
    extension: path.extname(resolvedPath).toLowerCase(),
    sizeBytes: stats.size,
    isImage: isImageFile(resolvedPath)
  };
}

function readFileAsText(filePath, encoding = 'utf8') {
  const resolvedPath = resolveFilePath(filePath);
  return fs.readFileSync(resolvedPath, { encoding });
}

function readFileAsBuffer(filePath) {
  const resolvedPath = resolveFilePath(filePath);
  return fs.readFileSync(resolvedPath);
}

module.exports = {
  isImageFile,
  summarizeFile,
  readFileAsText,
  readFileAsBuffer
};
