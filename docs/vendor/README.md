# ONNX Runtime Web

These unmodified runtime assets are from the official npm package
[`onnxruntime-web@1.31.1`](https://www.npmjs.com/package/onnxruntime-web/v/1.31.1).
The package is published by the [Microsoft ONNX Runtime project](https://github.com/microsoft/onnxruntime).

Included assets:

- `ort.wasm.min.js`: WebAssembly-only JavaScript browser runtime (avoids the JSEP/WebGPU bundle).
- `ort-wasm-simd-threaded.mjs`: WebAssembly loader.
- `ort-wasm-simd-threaded.wasm`: CPU WebAssembly runtime, also supports single-thread operation.
- `LICENSE.onnxruntime.txt`: upstream MIT license, copied from the `v1.31.1` release tag.
- `ThirdPartyNotices.txt`: upstream notices for bundled dependencies from the same release tag.

The application uses `executionProviders: ['wasm']` and `numThreads: 1`.
Serving these files from the same GitHub Pages site avoids a runtime dependency on an external CDN.

Acquisition: `npm pack onnxruntime-web@1.31.1`, then copy the three matching files from `package/dist/`.
Original npm tarball integrity:

```text
sha512-5GeH+bWQ+HxM10z7erGlsWqryrwMbSvm/1tkIsKTT48g0fI19VXVVC435t4t7VsPdSQeS7NmuZPXrQ73k82URw==
```

This directory contains third-party code under its own license.
