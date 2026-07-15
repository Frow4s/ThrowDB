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

class DescribeResponse(_message.Message):
    __slots__ = ("format_version", "schema", "record_size", "count", "capabilities")
    FORMAT_VERSION_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    RECORD_SIZE_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    format_version: int
    schema: CollectionSchema
    record_size: int
    count: int
    capabilities: Capabilities
    def __init__(self, format_version: _Optional[int] = ..., schema: _Optional[_Union[CollectionSchema, _Mapping]] = ..., record_size: _Optional[int] = ..., count: _Optional[int] = ..., capabilities: _Optional[_Union[Capabilities, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("operation_id", "ids", "embeddings")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    operation_id: OperationId
    ids: IdBatch
    embeddings: EmbeddingBatch
    def __init__(self, operation_id: _Optional[_Union[OperationId, _Mapping]] = ..., ids: _Optional[_Union[IdBatch, _Mapping]] = ..., embeddings: _Optional[_Union[EmbeddingBatch, _Mapping]] = ...) -> None: ...

class MutationResponse(_message.Message):
    __slots__ = ("committed_sequence", "affected_count")
    COMMITTED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    AFFECTED_COUNT_FIELD_NUMBER: _ClassVar[int]
    committed_sequence: int
    affected_count: int
    def __init__(self, committed_sequence: _Optional[int] = ..., affected_count: _Optional[int] = ...) -> None: ...

class DeleteRequest(_message.Message):
    __slots__ = ("operation_id", "ids")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    operation_id: OperationId
    ids: IdBatch
    def __init__(self, operation_id: _Optional[_Union[OperationId, _Mapping]] = ..., ids: _Optional[_Union[IdBatch, _Mapping]] = ...) -> None: ...

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
