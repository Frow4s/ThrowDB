# Build from the repository root after placing a Linux x86_64 wheel at
# wheelhouse/throwdb.whl. See docs/release-packaging.md.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG THROWDB_UID=10001
ARG THROWDB_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    TMPDIR=/tmp

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${THROWDB_GID}" throwdb \
    && useradd --uid "${THROWDB_UID}" --gid "${THROWDB_GID}" \
        --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin \
        throwdb \
    && mkdir -p /opt/throwdb /var/lib/throwdb \
    && chown "${THROWDB_UID}:${THROWDB_GID}" /var/lib/throwdb

COPY wheelhouse/throwdb.whl /tmp/throwdb.whl
RUN python -m pip install --no-cache-dir /tmp/throwdb.whl \
    && python -m pip check \
    && rm /tmp/throwdb.whl

COPY deploy/kubernetes/healthcheck.py /opt/throwdb/healthcheck.py
RUN chmod 0555 /opt/throwdb/healthcheck.py

USER ${THROWDB_UID}:${THROWDB_GID}
EXPOSE 50051

ENTRYPOINT ["throwdbd"]
CMD ["--help"]
