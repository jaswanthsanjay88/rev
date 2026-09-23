const fs = require('fs');
const path = require('path');

const distDir = path.join(__dirname, 'dist');
const cjsFile = path.join(distDir, 'index.js');
const mjsFile = path.join(distDir, 'index.mjs');

if (fs.existsSync(cjsFile)) {
  const cjsCode = fs.readFileSync(cjsFile, 'utf8');
  // Simple wrapper or dual build
  const esmCode = `
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const cjs = require('./index.js');

export const RevClient = cjs.RevClient;
export const rev = cjs.rev;
export const presets = cjs.presets;
export default cjs.rev;
`;
  fs.writeFileSync(mjsFile, esmCode.trim() + '\n', 'utf8');
  console.log('[+] Dual ESM & CJS bundles created in dist/');
}
