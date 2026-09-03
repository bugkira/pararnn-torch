Do not vendor Apple's code into `src/`. Clone read-only if you need the reference:

```bash
git clone --depth 1 https://github.com/apple/ml-pararnn.git third_party/ml-pararnn
```

License: Apple custom (not MIT). Notes: `docs/apple-ml-pararnn.md`. The clone is gitignored.
