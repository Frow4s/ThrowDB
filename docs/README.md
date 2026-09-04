# ThrowDB documentation

Start here according to the task:

| Goal | Document |
|---|---|
| Review the latest code changes, tradeoffs, and validation | [Refactoring review](refactoring-review.md) |
| Understand what is implemented and what is not | [Repository and production-readiness review](repository-review.md) |
| Create, read, mutate, tune, serve, back up, and restore | [End-user guide](user-guide.md) |
| Deploy one collection on Kubernetes/local NVMe | [Kubernetes deployment and operations](kubernetes-deployment.md) |
| Find/build/verify a wheel and runtime image | [Packaging and release artifacts](release-packaging.md) |
| Operate the gRPC protocol and daemon | [gRPC server contract](grpc-server.md) |
| Understand format-7 durability and maintenance | [Mutable storage](mutable-storage.md) |
| Understand tuning profile matching | [Reader tuning profiles](tuning-profiles.md) |
| Run benchmarks and acceptance tests | [Testing](testing.md) |

Format references:

- [Collection contract](collection-contract.md)
- [Format 7 mutable lifecycle](mutable-storage.md)
- [Formats 3–6 native layout](native-format-v3.md)
- [Older immutable format](storage-format.md)
- [Numeric format-v2 representation](storage-format-v2.md)
- [Product roadmap](product-roadmap.md)

The operational guides describe the current single-collection product. They
do not promise multi-node HA, online replication, UTF-8 IDs, or a
multi-collection database.
