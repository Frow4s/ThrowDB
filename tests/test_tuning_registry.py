from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import throwdb._tuning as tuning_module
from throwdb._tuning import (
    CANONICAL_READER_OPTIONS,
    READER_DEFAULTS,
    RegistryKey,
    TuningRegistryError,
    create_registry,
    list_profiles,
    load_registry,
    normalize_reader_options,
    remove_profile,
    reset_registry,
    select_profile,
    upsert_profile,
)


def _identity(device: int = 1) -> tuple[dict[str, object], ...]:
    return (
        {"kind": "local_block", "device": device, "serial": f"disk-{device}"},
        {"kernel": "6.1", "cpu_quota": 10},
        {"format": 6, "record_size": 2048},
    )


def test_reader_options_normalize_aliases_and_persisted_exactness() -> None:
    manual = normalize_reader_options(
        {
            "backend": "io_uring",
            "workers": np.int64(2),
            "queue_depth": 32,
            "read_range_bytes": 8192,
            "gap_bytes": 64,
        },
        record_size=2048,
    )
    assert set(manual) == CANONICAL_READER_OPTIONS
    assert manual == {
        **READER_DEFAULTS,
        "max_read_range_bytes": 8192,
        "max_gap_bytes": 64,
        "io_backend": "io_uring",
        "io_uring_workers": 2,
        "io_uring_queue_depth": 32,
    }
    assert normalize_reader_options(manual, persisted=True, record_size=2048) == manual

    with pytest.raises(ValueError, match="exactly the canonical"):
        normalize_reader_options({"io_workers": 1}, persisted=True)
    with pytest.raises(ValueError, match="unknown tuning option"):
        normalize_reader_options({"mystery": 1})
    for boolean in (True, np.bool_(True)):
        with pytest.raises(TypeError, match="integer"):
            normalize_reader_options({"io_workers": boolean})
    with pytest.raises(ValueError, match="max_read_range_bytes"):
        normalize_reader_options(
            {"max_read_range_bytes": 1024}, record_size=2048
        )
    with pytest.raises(ValueError, match="both be 0"):
        normalize_reader_options(
            {"backend": "io_uring", "io_workers": 1}
        )
    with pytest.raises(ValueError, match="queue_capacity"):
        normalize_reader_options({"io_queue_capacity": 1})


def test_missing_load_is_read_only_and_create_round_trips(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()

    missing = load_registry(store)
    assert missing.status == "missing"
    assert missing.registry is None
    assert missing.error is None
    assert not (store / "tuning").exists()

    created = create_registry(store)
    assert created.version == 1
    assert created.generation == 1
    assert len(created.salt) == 32
    assert created.salt == created.salt.lower()
    assert bytes.fromhex(created.salt)
    assert created.profiles == ()
    assert (store / "tuning" / "reader-profiles.lock").is_file()

    loaded = load_registry(store, strict=True)
    assert loaded.status == "ok"
    assert loaded.registry == created
    assert create_registry(store) == created


def test_upsert_exact_select_update_list_and_remove(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    device, runtime, layout = _identity()
    requested = {"io_workers": 2, "io_queue_capacity": 4}

    first = upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options=requested,
        source="manual",
        timestamp_ns=100,
        record_size=2048,
    )
    assert first.options == {
        **READER_DEFAULTS,
        "io_workers": 2,
        "io_queue_capacity": 4,
    }
    assert first.created_at_ns == first.updated_at_ns == 100
    assert first.source == "manual"
    assert dict(first.device) != device
    assert all(
        isinstance(value, str) and value.startswith("sha256:")
        for value in first.device.values()
    )
    assert requested == {"io_workers": 2, "io_queue_capacity": 4}
    assert first.as_dict()["key"] == first.key.as_dict()

    selected = select_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        record_size=2048,
    )
    assert selected.status == "selected"
    assert selected.key == first.key
    assert selected.profile == first

    moved_device, _, _ = _identity(2)
    no_match = select_profile(
        store,
        device=moved_device,
        runtime=runtime,
        layout=layout,
        record_size=2048,
    )
    assert no_match.status == "no_match"
    assert no_match.key != first.key

    second = upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options={"io_workers": 3},
        source="calibrated",
        timestamp_ns=200,
        record_size=2048,
    )
    assert second.key == first.key
    assert second.created_at_ns == 100
    assert second.updated_at_ns == 200
    assert second.options["io_workers"] == 3
    profiles = list_profiles(store)
    assert profiles == [second]

    # Public mappings are immutable and as_dict() returns detached dictionaries.
    with pytest.raises(TypeError):
        second.options["io_workers"] = 99  # type: ignore[index]
    detached = second.as_dict()
    detached_options = detached["options"]
    assert isinstance(detached_options, dict)
    detached_options["io_workers"] = 99
    assert list_profiles(store)[0].options["io_workers"] == 3

    assert remove_profile(store, RegistryKey(**second.key.as_dict())) is True
    assert list_profiles(store) == []
    assert remove_profile(store, second.key) is False


def test_checksum_duplicate_keys_and_corruption_diagnostics(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    device, runtime, layout = _identity()
    upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options={},
    )
    path = store / "tuning" / "reader-profiles.json"

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["generation"] += 1
    path.write_text(json.dumps(envelope), encoding="utf-8")
    damaged = load_registry(store)
    assert damaged.status == "corrupt"
    assert damaged.registry is None
    assert "checksum mismatch" in (damaged.error or "")
    with pytest.raises(TuningRegistryError, match="checksum mismatch"):
        load_registry(store, strict=True)
    with pytest.raises(TuningRegistryError, match="checksum mismatch"):
        list_profiles(store)

    path.write_text(
        '{"format":"throwdb-reader-profiles",'
        '"format":"throwdb-reader-profiles"}',
        encoding="utf-8",
    )
    duplicate = load_registry(store)
    assert duplicate.status == "corrupt"
    assert "duplicate key" in (duplicate.error or "")
    with pytest.raises(TuningRegistryError, match="force=True"):
        reset_registry(store)
    assert reset_registry(store, force=True) is True
    assert load_registry(store).status == "missing"
    assert reset_registry(store, force=True) is False


def test_excessively_nested_registry_is_reported_as_corrupt(tmp_path: Path) -> None:
    store = tmp_path / "store"
    tuning = store / "tuning"
    tuning.mkdir(parents=True)
    path = tuning / "reader-profiles.json"
    path.write_text("[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")

    loaded = load_registry(store)
    assert loaded.status == "corrupt"
    assert loaded.registry is None
    assert "decode" in (loaded.error or "")
    with pytest.raises(TuningRegistryError, match="decode"):
        load_registry(store, strict=True)


def test_reset_valid_registry_is_atomic_and_allows_a_new_salt(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    original = create_registry(store)
    assert reset_registry(store) is True
    assert load_registry(store).status == "missing"
    replacement = create_registry(store)
    assert replacement.salt != original.salt
    assert not list((store / "tuning").glob("*.quarantine.*"))


def test_registry_never_follows_a_tuning_directory_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    create_registry(outside)

    store = tmp_path / "store"
    store.mkdir()
    (store / "tuning").symlink_to(outside / "tuning", target_is_directory=True)

    loaded = load_registry(store)
    assert loaded.status == "corrupt"
    assert "not a real directory" in (loaded.error or "")
    with pytest.raises(TuningRegistryError, match="not a real directory"):
        create_registry(store)


def test_registry_publication_stays_on_anchored_directory_during_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    outside = tmp_path / "outside"
    store.mkdir()
    outside.mkdir()
    create_registry(store)
    sentinel = b"outside must not be replaced"
    (outside / "reader-profiles.json").write_bytes(sentinel)

    original_encode = tuning_module._encode_registry
    detached = store / "detached-tuning"
    swapped = False

    def swap_then_encode(registry: tuning_module.TuningRegistry) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            (store / "tuning").rename(detached)
            (store / "tuning").symlink_to(outside, target_is_directory=True)
        return original_encode(registry)

    monkeypatch.setattr(tuning_module, "_encode_registry", swap_then_encode)
    device, runtime, layout = _identity()
    profile = upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options={"io_workers": 2},
    )

    assert (outside / "reader-profiles.json").read_bytes() == sentinel
    assert not list(outside.glob("reader-profiles.*.tmp"))
    (store / "tuning").unlink()
    detached.rename(store / "tuning")
    assert list_profiles(store) == [profile]


def test_registry_reset_stays_on_anchored_directory_during_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    outside = tmp_path / "outside"
    store.mkdir()
    outside.mkdir()
    create_registry(store)
    sentinel = b"outside must not be removed"
    (outside / "reader-profiles.json").write_bytes(sentinel)

    original_load = tuning_module._load_registry_from_directory
    detached = store / "detached-tuning"
    swapped = False

    def load_then_swap(
        tuning_descriptor: int,
        *,
        strict: bool,
    ) -> tuning_module.RegistryLoadResult:
        nonlocal swapped
        result = original_load(tuning_descriptor, strict=strict)
        if not swapped:
            swapped = True
            (store / "tuning").rename(detached)
            (store / "tuning").symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(
        tuning_module,
        "_load_registry_from_directory",
        load_then_swap,
    )
    assert reset_registry(store) is True

    assert (outside / "reader-profiles.json").read_bytes() == sentinel
    (store / "tuning").unlink()
    detached.rename(store / "tuning")
    assert load_registry(store).status == "missing"
    assert not list((store / "tuning").glob("*.quarantine.*"))


def test_upsert_rejects_a_registry_recreated_after_identity_probe(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    original = create_registry(store)
    assert reset_registry(store)
    replacement = create_registry(store)
    assert replacement.salt != original.salt

    device, runtime, layout = _identity()
    with pytest.raises(TuningRegistryError, match="identity changed"):
        upsert_profile(
            store,
            device=device,
            runtime=runtime,
            layout=layout,
            options={},
            expected_salt=original.salt,
        )
    assert list_profiles(store) == []


def test_upsert_generation_compare_and_swap_succeeds(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    registry = create_registry(store)
    device, runtime, layout = _identity()

    profile = upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options={"io_workers": 2},
        expected_generation=registry.generation,
    )

    loaded = load_registry(store, strict=True)
    assert loaded.registry is not None
    assert loaded.registry.generation == registry.generation + 1
    assert loaded.registry.profiles == (profile,)


def test_upsert_stale_generation_does_not_change_registry_bytes(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    device, runtime, layout = _identity()
    upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options={"io_workers": 1},
    )
    registry_path = store / "tuning" / "reader-profiles.json"
    before = registry_path.read_bytes()

    with pytest.raises(TuningRegistryError, match="generation changed"):
        upsert_profile(
            store,
            device=device,
            runtime=runtime,
            layout=layout,
            options={"io_workers": 2},
            expected_generation=0,
        )

    assert registry_path.read_bytes() == before


def test_upsert_bootstraps_with_initial_salt_in_one_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    device, runtime, layout = _identity()
    initial_salt = "01" * 16
    publications: list[tuning_module.TuningRegistry] = []
    original_publish = tuning_module._publish_registry

    def record_publish(
        tuning_descriptor: int,
        registry: tuning_module.TuningRegistry,
    ) -> None:
        publications.append(registry)
        original_publish(tuning_descriptor, registry)

    monkeypatch.setattr(tuning_module, "_publish_registry", record_publish)
    profile = upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options={},
        expected_salt=initial_salt,
        expected_generation=0,
        initial_salt=initial_salt,
    )

    assert len(publications) == 1
    assert publications[0].generation == 1
    assert publications[0].salt == initial_salt
    assert publications[0].profiles == (profile,)
    loaded = load_registry(store, strict=True)
    assert loaded.registry == publications[0]


def test_upsert_rechecks_anchored_collection_root_before_creating_metadata(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    status = store.stat()
    device, runtime, layout = _identity()

    with pytest.raises(TuningRegistryError, match="collection root changed"):
        upsert_profile(
            store,
            device=device,
            runtime=runtime,
            layout=layout,
            options={},
            initial_salt="01" * 16,
            expected_store_identity=(status.st_dev, status.st_ino + 1),
        )

    assert not (store / "tuning").exists()


def test_calibration_publication_uses_a_fail_fast_registry_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    device, runtime, layout = _identity()
    original_flock = tuning_module.fcntl.flock

    def busy_on_nonblocking(descriptor: int, operation: int) -> None:
        if operation & tuning_module.fcntl.LOCK_NB:
            raise BlockingIOError("fixture lock contention")
        original_flock(descriptor, operation)

    monkeypatch.setattr(tuning_module.fcntl, "flock", busy_on_nonblocking)

    with pytest.raises(TuningRegistryError, match="registry is busy"):
        upsert_profile(
            store,
            device=device,
            runtime=runtime,
            layout=layout,
            options={},
            fail_if_locked=True,
        )

    assert load_registry(store).status == "missing"


def test_fail_if_locked_requires_a_bool(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    device, runtime, layout = _identity()

    with pytest.raises(TypeError, match="fail_if_locked"):
        upsert_profile(
            store,
            device=device,
            runtime=runtime,
            layout=layout,
            options={},
            fail_if_locked=1,  # type: ignore[arg-type]
        )


def test_implicit_timestamp_does_not_regress_after_clock_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    device, runtime, layout = _identity()
    first = upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options={"io_workers": 1},
        timestamp_ns=200,
    )
    assert first.created_at_ns == first.updated_at_ns == 200

    monkeypatch.setattr(tuning_module.time, "time_ns", lambda: 100)
    second = upsert_profile(
        store,
        device=device,
        runtime=runtime,
        layout=layout,
        options={"io_workers": 2},
    )
    assert second.created_at_ns == 200
    assert second.updated_at_ns == 200
    assert second.options["io_workers"] == 2

    with pytest.raises(ValueError, match="predates"):
        upsert_profile(
            store,
            device=device,
            runtime=runtime,
            layout=layout,
            options={},
            timestamp_ns=199,
        )


def test_concurrent_writers_merge_distinct_exact_keys(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()

    def save(index: int) -> RegistryKey:
        device, runtime, layout = _identity(index)
        profile = upsert_profile(
            store,
            device=device,
            runtime=runtime,
            layout=layout,
            options={"io_workers": (index % 4) + 1},
            source="concurrent-test",
            timestamp_ns=1000 + index,
            record_size=2048,
        )
        return profile.key

    with ThreadPoolExecutor(max_workers=8) as executor:
        keys = list(executor.map(save, range(1, 17)))

    profiles = list_profiles(store)
    assert len(profiles) == 16
    assert {profile.key for profile in profiles} == set(keys)
    loaded = load_registry(store, strict=True)
    assert loaded.registry is not None
    assert loaded.registry.generation == 16
    assert not list((store / "tuning").glob("*.tmp"))
