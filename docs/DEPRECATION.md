# Namespace migration

OpenLocalAgent is the supported project and Python package name. The former `localagent` package
has been retired and is no longer included by the build configuration. New code must import from
`openlocalagent` and use the `openlocalagent` console entry points.

## Migration

```text
from localagent...           -> from openlocalagent...
import localagent...         -> import openlocalagent...
python -m localagent...      -> python -m openlocalagent...
```

The repository keeps historical artifact identifiers such as `localagent_v1` and
`localagent_*_manifest` unchanged. They are serialized data-format and provenance identifiers,
not Python namespaces; changing them would invalidate existing manifests and evaluation receipts.

Published Hugging Face model and Space names that contain `localagent` are historical release
coordinates. New releases should use `openlocalagent` names, while older links remain documented
where they are needed to reproduce a past result.

The checked-in `spaces/localagent-webgpu` bundle and `localagent_*` benchmark/receipt identifiers
are likewise frozen compatibility coordinates. The active Python package, CLI, exporters, and new
release examples use `openlocalagent`; the frozen bundle is not imported as a Python package.
