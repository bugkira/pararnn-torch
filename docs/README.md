# Docs index (maintainer)

Public site nav is defined in `/mkdocs.yml`. Tree:

```text
docs/
  getting_started/   # quickstart, adoption, shapes
  core/              # newton_scan, jacobian_classes, numerics, backward_scan
  cells/             # catalog by family + fidelity + YAML specs
    classic/ normalized/ matrix/ continuous/
  systems/           # inference, vLLM, distributed, compile, OOM
  audit/             # verification hub, xLSTM notes, repo layout
```

Cell specs use **Diff + Strict Spec Contract** ([spec_template.md](cells/spec_template.md)).
