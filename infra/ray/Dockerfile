# ---- Stage 1: build py-spy for ARM64 (Apple Silicon) ----
FROM --platform=linux/arm64 rust:1.78-bullseye AS builder

# Build a static-ish py-spy binary and place in /out/bin
RUN cargo install py-spy --version 0.3.14 --locked --root /out

# ---- Stage 2: Ray runtime with py-spy installed ----
FROM rayproject/ray:2.35.0-py311-aarch64

USER root

# Copy the py-spy binary from the builder
COPY --from=builder /out/bin/py-spy /usr/local/bin/py-spy

# Make py-spy setuid root; create a sudo shim; add symlink where Ray looks
RUN set -eux; \
    PYSPY_BIN="/usr/local/bin/py-spy"; \
    test -x "${PYSPY_BIN}"; \
    chown root:root "${PYSPY_BIN}"; \
    chmod u+s "${PYSPY_BIN}"; \
    # Minimal sudo shim so 'sudo -n <cmd>' just runs <cmd> (no password)
    printf '%s\n' '#!/bin/sh' \
                  'while [ "$1" != "" ] && [ "${1#-}" != "$1" ]; do shift; done' \
                  'exec "$@"' > /usr/bin/sudo; \
    chmod 0755 /usr/bin/sudo; \
    # Ray sometimes invokes this exact path
    mkdir -p /home/ray/anaconda3/bin; \
    ln -sf "${PYSPY_BIN}" /home/ray/anaconda3/bin/py-spy; \
    # Show perms for sanity
    ls -l "${PYSPY_BIN}" /usr/bin/sudo

# Drop privileges (setuid py-spy will still escalate for profiling)
USER ray
