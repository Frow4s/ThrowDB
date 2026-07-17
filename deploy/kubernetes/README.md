# Kubernetes assets

The `base/` directory is a reviewed Kustomize starting point for one writable
format-7 collection on a static local-NVMe PVC. It expects:

- an image named `throwdb:0.1.0` to be replaced by an immutable registry
  digest;
- a headless governing Service plus a client-facing ClusterIP Service;
- PVC `throwdb-data` backed by storage class `throwdb-local-nvme`;
- Secrets `throwdb-server-tls` and `throwdb-probe-client-tls`;
- collection path `/var/lib/throwdb/collection`;
- exactly one replica.

Do not apply the examples unchanged. They contain explicit `REPLACE_*`
placeholders and offline Jobs that must never overlap the writable server.

Full instructions, TLS requirements, backup downtime, restore, monitoring,
updates, and failure runbooks are in
[`docs/kubernetes-deployment.md`](../../docs/kubernetes-deployment.md).
