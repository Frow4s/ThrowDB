from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class IdType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ID_TYPE_UNSPECIFIED: _ClassVar[IdType]
    ID_TYPE_UINT64: _ClassVar[IdType]
    ID_TYPE_UTF8: _ClassVar[IdType]

class EmbeddingDType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EMBEDDING_DTYPE_UNSPECIFIED: _ClassVar[EmbeddingDType]
    EMBEDDING_DTYPE_FLOAT16: _ClassVar[EmbeddingDType]
    EMBEDDING_DTYPE_FLOAT32: _ClassVar[EmbeddingDType]

class ReaderBackend(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    READER_BACKEND_UNSPECIFIED: _ClassVar[ReaderBackend]
    READER_BACKEND_THREAD_POOL: _ClassVar[ReaderBackend]
    READER_BACKEND_IO_URING: _ClassVar[ReaderBackend]

class LocationEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOCATION_ENCODING_UNSPECIFIED: _ClassVar[LocationEncoding]
    LOCATION_ENCODING_SEGMENT_ROW_U32_LE: _ClassVar[LocationEncoding]
ID_TYPE_UNSPECIFIED: IdType
ID_TYPE_UINT64: IdType
ID_TYPE_UTF8: IdType
EMBEDDING_DTYPE_UNSPECIFIED: EmbeddingDType
EMBEDDING_DTYPE_FLOAT16: EmbeddingDType
EMBEDDING_DTYPE_FLOAT32: EmbeddingDType
READER_BACKEND_UNSPECIFIED: ReaderBackend
READER_BACKEND_THREAD_POOL: ReaderBackend
READER_BACKEND_IO_URING: ReaderBackend
LOCATION_ENCODING_UNSPECIFIED: LocationEncoding
LOCATION_ENCODING_SEGMENT_ROW_U32_LE: LocationEncoding

class UInt64Ids(_message.Message):
    __slots__ = ("count", "values_le")
    COUNT_FIELD_NUMBER: _ClassVar[int]
    VALUES_LE_FIELD_NUMBER: _ClassVar[int]
    count: int
    values_le: bytes
    def __init__(self, count: _Optional[int] = ..., values_le: _Optional[bytes] = ...) -> None: ...

class Utf8Ids(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...

class IdBatch(_message.Message):
    __slots__ = ("uint64_ids", "utf8_ids")
    UINT64_IDS_FIELD_NUMBER: _ClassVar[int]
    UTF8_IDS_FIELD_NUMBER: _ClassVar[int]
    uint64_ids: UInt64Ids
    utf8_ids: Utf8Ids
    def __init__(self, uint64_ids: _Optional[_Union[UInt64Ids, _Mapping]] = ..., utf8_ids: _Optional[_Union[Utf8Ids, _Mapping]] = ...) -> None: ...

class EmbeddingBatch(_message.Message):
    __slots__ = ("dtype", "dimension", "row_count", "values_le")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    VALUES_LE_FIELD_NUMBER: _ClassVar[int]
    dtype: EmbeddingDType
    dimension: int
    row_count: int
    values_le: bytes
    def __init__(self, dtype: _Optional[_Union[EmbeddingDType, str]] = ..., dimension: _Optional[int] = ..., row_count: _Optional[int] = ..., values_le: _Optional[bytes] = ...) -> None: ...

class OperationId(_message.Message):
    __slots__ = ("high", "low")
    HIGH_FIELD_NUMBER: _ClassVar[int]
    LOW_FIELD_NUMBER: _ClassVar[int]
    high: int
    low: int
    def __init__(self, high: _Optional[int] = ..., low: _Optional[int] = ...) -> None: ...

class CollectionId(_message.Message):
    __slots__ = ("low", "high")
    LOW_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    low: int
    high: int
    def __init__(self, low: _Optional[int] = ..., high: _Optional[int] = ...) -> None: ...

class ScanToken(_message.Message):
    __slots__ = ("high", "low")
    HIGH_FIELD_NUMBER: _ClassVar[int]
    LOW_FIELD_NUMBER: _ClassVar[int]
    high: int
    low: int
    def __init__(self, high: _Optional[int] = ..., low: _Optional[int] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "protocol_major", "protocol_minor")
    class ServingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SERVING_STATUS_UNSPECIFIED: _ClassVar[HealthResponse.ServingStatus]
        SERVING_STATUS_SERVING: _ClassVar[HealthResponse.ServingStatus]
        SERVING_STATUS_NOT_SERVING: _ClassVar[HealthResponse.ServingStatus]
    SERVING_STATUS_UNSPECIFIED: HealthResponse.ServingStatus
    SERVING_STATUS_SERVING: HealthResponse.ServingStatus
    SERVING_STATUS_NOT_SERVING: HealthResponse.ServingStatus
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_MAJOR_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_MINOR_FIELD_NUMBER: _ClassVar[int]
    status: HealthResponse.ServingStatus
    protocol_major: int
    protocol_minor: int
    def __init__(self, status: _Optional[_Union[HealthResponse.ServingStatus, str]] = ..., protocol_major: _Optional[int] = ..., protocol_minor: _Optional[int] = ...) -> None: ...

class DescribeRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Capabilities(_message.Message):
    __slots__ = ("writable", "prefix_get", "scan_locations", "durable_idempotency")
    WRITABLE_FIELD_NUMBER: _ClassVar[int]
    PREFIX_GET_FIELD_NUMBER: _ClassVar[int]
    SCAN_LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    DURABLE_IDEMPOTENCY_FIELD_NUMBER: _ClassVar[int]
    writable: bool
    prefix_get: bool
    scan_locations: bool
    durable_idempotency: bool
    def __init__(self, writable: _Optional[bool] = ..., prefix_get: _Optional[bool] = ..., scan_locations: _Optional[bool] = ..., durable_idempotency: _Optional[bool] = ...) -> None: ...

class CollectionSchema(_message.Message):
    __slots__ = ("id_type", "embedding_dtype", "dimension")
    ID_TYPE_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DTYPE_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    id_type: IdType
    embedding_dtype: EmbeddingDType
    dimension: int
    def __init__(self, id_type: _Optional[_Union[IdType, str]] = ..., embedding_dtype: _Optional[_Union[EmbeddingDType, str]] = ..., dimension: _Optional[int] = ...) -> None: ...

class ReaderConfiguration(_message.Message):
    __slots__ = ("max_read_range_bytes", "max_gap_bytes", "io_workers", "io_queue_capacity", "io_ranges_per_task", "io_backend", "io_uring_queue_depth", "io_uring_workers")
    MAX_READ_RANGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_GAP_BYTES_FIELD_NUMBER: _ClassVar[int]
    IO_WORKERS_FIELD_NUMBER: _ClassVar[int]
    IO_QUEUE_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    IO_RANGES_PER_TASK_FIELD_NUMBER: _ClassVar[int]
    IO_BACKEND_FIELD_NUMBER: _ClassVar[int]
    IO_URING_QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    IO_URING_WORKERS_FIELD_NUMBER: _ClassVar[int]
    max_read_range_bytes: int
    max_gap_bytes: int
    io_workers: int
    io_queue_capacity: int
    io_ranges_per_task: int
    io_backend: ReaderBackend
    io_uring_queue_depth: int
    io_uring_workers: int
    def __init__(self, max_read_range_bytes: _Optional[int] = ..., max_gap_bytes: _Optional[int] = ..., io_workers: _Optional[int] = ..., io_queue_capacity: _Optional[int] = ..., io_ranges_per_task: _Optional[int] = ..., io_backend: _Optional[_Union[ReaderBackend, str]] = ..., io_uring_queue_depth: _Optional[int] = ..., io_uring_workers: _Optional[int] = ...) -> None: ...

class TuningSelection(_message.Message):
    __slots__ = ("mode", "source", "status", "reuse_level", "explicit_overrides")
    MODE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REUSE_LEVEL_FIELD_NUMBER: _ClassVar[int]
    EXPLICIT_OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    mode: str
    source: str
    status: str
    reuse_level: str
    explicit_overrides: bool
    def __init__(self, mode: _Optional[str] = ..., source: _Optional[str] = ..., status: _Optional[str] = ..., reuse_level: _Optional[str] = ..., explicit_overrides: _Optional[bool] = ...) -> None: ...

class AvailabilityConfiguration(_message.Message):
    __slots__ = ("max_concurrent_calls", "max_concurrent_scans", "max_unary_inflight_bytes", "max_scan_inflight_bytes", "max_grpc_memory_bytes", "scan_max_duration_millis", "scan_idle_timeout_millis", "shutdown_grace_millis", "shutdown_hard_timeout_millis")
    MAX_CONCURRENT_CALLS_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENT_SCANS_FIELD_NUMBER: _ClassVar[int]
    MAX_UNARY_INFLIGHT_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_SCAN_INFLIGHT_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_GRPC_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    SCAN_MAX_DURATION_MILLIS_FIELD_NUMBER: _ClassVar[int]
    SCAN_IDLE_TIMEOUT_MILLIS_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_GRACE_MILLIS_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_HARD_TIMEOUT_MILLIS_FIELD_NUMBER: _ClassVar[int]
    max_concurrent_calls: int
    max_concurrent_scans: int
    max_unary_inflight_bytes: int
    max_scan_inflight_bytes: int
    max_grpc_memory_bytes: int
    scan_max_duration_millis: int
    scan_idle_timeout_millis: int
    shutdown_grace_millis: int
    shutdown_hard_timeout_millis: int
    def __init__(self, max_concurrent_calls: _Optional[int] = ..., max_concurrent_scans: _Optional[int] = ..., max_unary_inflight_bytes: _Optional[int] = ..., max_scan_inflight_bytes: _Optional[int] = ..., max_grpc_memory_bytes: _Optional[int] = ..., scan_max_duration_millis: _Optional[int] = ..., scan_idle_timeout_millis: _Optional[int] = ..., shutdown_grace_millis: _Optional[int] = ..., shutdown_hard_timeout_millis: _Optional[int] = ...) -> None: ...

class AdmissionStatistics(_message.Message):
    __slots__ = ("active_unary_calls", "active_scans", "unary_estimated_bytes", "scan_estimated_bytes", "admitted_unary_calls", "rejected_unary_calls", "admitted_scans", "rejected_scans", "admission_closed", "draining", "terminal_faulted")
    ACTIVE_UNARY_CALLS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SCANS_FIELD_NUMBER: _ClassVar[int]
    UNARY_ESTIMATED_BYTES_FIELD_NUMBER: _ClassVar[int]
    SCAN_ESTIMATED_BYTES_FIELD_NUMBER: _ClassVar[int]
    ADMITTED_UNARY_CALLS_FIELD_NUMBER: _ClassVar[int]
    REJECTED_UNARY_CALLS_FIELD_NUMBER: _ClassVar[int]
    ADMITTED_SCANS_FIELD_NUMBER: _ClassVar[int]
    REJECTED_SCANS_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_CLOSED_FIELD_NUMBER: _ClassVar[int]
    DRAINING_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_FAULTED_FIELD_NUMBER: _ClassVar[int]
    active_unary_calls: int
    active_scans: int
    unary_estimated_bytes: int
    scan_estimated_bytes: int
    admitted_unary_calls: int
    rejected_unary_calls: int
    admitted_scans: int
    rejected_scans: int
    admission_closed: bool
    draining: bool
    terminal_faulted: bool
    def __init__(self, active_unary_calls: _Optional[int] = ..., active_scans: _Optional[int] = ..., unary_estimated_bytes: _Optional[int] = ..., scan_estimated_bytes: _Optional[int] = ..., admitted_unary_calls: _Optional[int] = ..., rejected_unary_calls: _Optional[int] = ..., admitted_scans: _Optional[int] = ..., rejected_scans: _Optional[int] = ..., admission_closed: _Optional[bool] = ..., draining: _Optional[bool] = ..., terminal_faulted: _Optional[bool] = ...) -> None: ...

class EngineStatistics(_message.Message):
    __slots__ = ("catalog_generation", "committed_sequence", "materialized_sequence", "active_wal_bytes", "active_wal_transactions", "active_wal_operations", "frozen_wal_count", "frozen_wal_bytes", "total_wal_bytes", "total_wal_operations", "overlay_entries", "estimated_overlay_bytes", "delta_run_count", "total_delta_run_bytes", "total_delta_segment_bytes", "idempotency_journal_bytes", "idempotency_completed_operations", "idempotency_aborted_operations", "idempotency_max_bytes", "idempotency_max_operations", "idempotency_generation", "filesystem_available_bytes", "mutation_disk_reserve_bytes", "maintenance_active")
    CATALOG_GENERATION_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    MATERIALIZED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_WAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_WAL_TRANSACTIONS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_WAL_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    FROZEN_WAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    FROZEN_WAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_WAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_WAL_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    OVERLAY_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_OVERLAY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DELTA_RUN_COUNT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DELTA_RUN_BYTES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DELTA_SEGMENT_BYTES_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_JOURNAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_COMPLETED_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_ABORTED_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_MAX_OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_GENERATION_FIELD_NUMBER: _ClassVar[int]
    FILESYSTEM_AVAILABLE_BYTES_FIELD_NUMBER: _ClassVar[int]
    MUTATION_DISK_RESERVE_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAINTENANCE_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    catalog_generation: int
    committed_sequence: int
    materialized_sequence: int
    active_wal_bytes: int
    active_wal_transactions: int
    active_wal_operations: int
    frozen_wal_count: int
    frozen_wal_bytes: int
    total_wal_bytes: int
    total_wal_operations: int
    overlay_entries: int
    estimated_overlay_bytes: int
    delta_run_count: int
    total_delta_run_bytes: int
    total_delta_segment_bytes: int
    idempotency_journal_bytes: int
    idempotency_completed_operations: int
    idempotency_aborted_operations: int
    idempotency_max_bytes: int
    idempotency_max_operations: int
    idempotency_generation: int
    filesystem_available_bytes: int
    mutation_disk_reserve_bytes: int
    maintenance_active: bool
    def __init__(self, catalog_generation: _Optional[int] = ..., committed_sequence: _Optional[int] = ..., materialized_sequence: _Optional[int] = ..., active_wal_bytes: _Optional[int] = ..., active_wal_transactions: _Optional[int] = ..., active_wal_operations: _Optional[int] = ..., frozen_wal_count: _Optional[int] = ..., frozen_wal_bytes: _Optional[int] = ..., total_wal_bytes: _Optional[int] = ..., total_wal_operations: _Optional[int] = ..., overlay_entries: _Optional[int] = ..., estimated_overlay_bytes: _Optional[int] = ..., delta_run_count: _Optional[int] = ..., total_delta_run_bytes: _Optional[int] = ..., total_delta_segment_bytes: _Optional[int] = ..., idempotency_journal_bytes: _Optional[int] = ..., idempotency_completed_operations: _Optional[int] = ..., idempotency_aborted_operations: _Optional[int] = ..., idempotency_max_bytes: _Optional[int] = ..., idempotency_max_operations: _Optional[int] = ..., idempotency_generation: _Optional[int] = ..., filesystem_available_bytes: _Optional[int] = ..., mutation_disk_reserve_bytes: _Optional[int] = ..., maintenance_active: _Optional[bool] = ...) -> None: ...

class DescribeResponse(_message.Message):
    __slots__ = ("format_version", "schema", "record_size", "count", "capabilities", "reader", "tuning", "availability", "idempotency_domain", "admission_stats", "engine_stats")
    FORMAT_VERSION_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    RECORD_SIZE_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    READER_FIELD_NUMBER: _ClassVar[int]
    TUNING_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    ADMISSION_STATS_FIELD_NUMBER: _ClassVar[int]
    ENGINE_STATS_FIELD_NUMBER: _ClassVar[int]
    format_version: int
    schema: CollectionSchema
    record_size: int
    count: int
    capabilities: Capabilities
    reader: ReaderConfiguration
    tuning: TuningSelection
    availability: AvailabilityConfiguration
    idempotency_domain: OperationId
    admission_stats: AdmissionStatistics
    engine_stats: EngineStatistics
    def __init__(self, format_version: _Optional[int] = ..., schema: _Optional[_Union[CollectionSchema, _Mapping]] = ..., record_size: _Optional[int] = ..., count: _Optional[int] = ..., capabilities: _Optional[_Union[Capabilities, _Mapping]] = ..., reader: _Optional[_Union[ReaderConfiguration, _Mapping]] = ..., tuning: _Optional[_Union[TuningSelection, _Mapping]] = ..., availability: _Optional[_Union[AvailabilityConfiguration, _Mapping]] = ..., idempotency_domain: _Optional[_Union[OperationId, _Mapping]] = ..., admission_stats: _Optional[_Union[AdmissionStatistics, _Mapping]] = ..., engine_stats: _Optional[_Union[EngineStatistics, _Mapping]] = ...) -> None: ...

class GetRequest(_message.Message):
    __slots__ = ("ids", "prefix_dimensions")
    IDS_FIELD_NUMBER: _ClassVar[int]
    PREFIX_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    ids: IdBatch
    prefix_dimensions: int
    def __init__(self, ids: _Optional[_Union[IdBatch, _Mapping]] = ..., prefix_dimensions: _Optional[int] = ...) -> None: ...

class GetResponse(_message.Message):
    __slots__ = ("embeddings",)
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    embeddings: EmbeddingBatch
    def __init__(self, embeddings: _Optional[_Union[EmbeddingBatch, _Mapping]] = ...) -> None: ...

class MutationRequest(_message.Message):
    __slots__ = ("operation_id", "ids", "embeddings", "idempotency_domain")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    operation_id: OperationId
    ids: IdBatch
    embeddings: EmbeddingBatch
    idempotency_domain: OperationId
    def __init__(self, operation_id: _Optional[_Union[OperationId, _Mapping]] = ..., ids: _Optional[_Union[IdBatch, _Mapping]] = ..., embeddings: _Optional[_Union[EmbeddingBatch, _Mapping]] = ..., idempotency_domain: _Optional[_Union[OperationId, _Mapping]] = ...) -> None: ...

class MutationResponse(_message.Message):
    __slots__ = ("committed_sequence", "affected_count")
    COMMITTED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    AFFECTED_COUNT_FIELD_NUMBER: _ClassVar[int]
    committed_sequence: int
    affected_count: int
    def __init__(self, committed_sequence: _Optional[int] = ..., affected_count: _Optional[int] = ...) -> None: ...

class DeleteRequest(_message.Message):
    __slots__ = ("operation_id", "ids", "idempotency_domain")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    operation_id: OperationId
    ids: IdBatch
    idempotency_domain: OperationId
    def __init__(self, operation_id: _Optional[_Union[OperationId, _Mapping]] = ..., ids: _Optional[_Union[IdBatch, _Mapping]] = ..., idempotency_domain: _Optional[_Union[OperationId, _Mapping]] = ...) -> None: ...

class DeleteResponse(_message.Message):
    __slots__ = ("committed_sequence", "affected_count", "deleted_mask")
    COMMITTED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    AFFECTED_COUNT_FIELD_NUMBER: _ClassVar[int]
    DELETED_MASK_FIELD_NUMBER: _ClassVar[int]
    committed_sequence: int
    affected_count: int
    deleted_mask: bytes
    def __init__(self, committed_sequence: _Optional[int] = ..., affected_count: _Optional[int] = ..., deleted_mask: _Optional[bytes] = ...) -> None: ...

class ScanIdsRequest(_message.Message):
    __slots__ = ("batch_rows", "include_locations")
    BATCH_ROWS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    batch_rows: int
    include_locations: bool
    def __init__(self, batch_rows: _Optional[int] = ..., include_locations: _Optional[bool] = ...) -> None: ...

class IdScanSnapshot(_message.Message):
    __slots__ = ("collection_id", "catalog_generation", "committed_sequence", "count", "token")
    COLLECTION_ID_FIELD_NUMBER: _ClassVar[int]
    CATALOG_GENERATION_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    collection_id: CollectionId
    catalog_generation: int
    committed_sequence: int
    count: int
    token: ScanToken
    def __init__(self, collection_id: _Optional[_Union[CollectionId, _Mapping]] = ..., catalog_generation: _Optional[int] = ..., committed_sequence: _Optional[int] = ..., count: _Optional[int] = ..., token: _Optional[_Union[ScanToken, _Mapping]] = ...) -> None: ...

class ScanIdsResponse(_message.Message):
    __slots__ = ("snapshot", "emitted_offset", "ids", "location_encoding", "locations_le")
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    EMITTED_OFFSET_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    LOCATION_ENCODING_FIELD_NUMBER: _ClassVar[int]
    LOCATIONS_LE_FIELD_NUMBER: _ClassVar[int]
    snapshot: IdScanSnapshot
    emitted_offset: int
    ids: IdBatch
    location_encoding: LocationEncoding
    locations_le: bytes
    def __init__(self, snapshot: _Optional[_Union[IdScanSnapshot, _Mapping]] = ..., emitted_offset: _Optional[int] = ..., ids: _Optional[_Union[IdBatch, _Mapping]] = ..., location_encoding: _Optional[_Union[LocationEncoding, str]] = ..., locations_le: _Optional[bytes] = ...) -> None: ...
