#!/bin/bash

set -e

DOC_STRING="Build the multi-architecture quick-evaluation image (SUMO only, no CARLA).

Without --push, builds for the host architecture only and loads it into the local
Docker image store as i24-sim-quick:latest. With --push, builds linux/amd64 and
linux/arm64 together and pushes the multi-arch manifest to REPO."

USAGE_STRING="Usage: $0 [--push] [--repo REPO] [-h|--help]

    --push         Build both architectures and push to REPO (requires docker login)
    --repo REPO    Registry repository to push to (default: papapalpi/i24-sim-quick)"

PUSH=0
REPO="papapalpi/i24-sim-quick"

# Plain argument loop: macOS ships BSD getopt, which has no --long support.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push ) PUSH=1; shift ;;
    --repo ) REPO="$2"; shift 2 ;;
    -h | --help ) echo "$DOC_STRING"; echo; echo "$USAGE_STRING"; exit 0 ;;
    * ) echo "Unknown option: $1"; echo "$USAGE_STRING"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $PUSH -eq 1 ]]; then
  # The default "docker" driver cannot build multi-platform images; use a
  # docker-container builder (created once, reused afterwards).
  docker buildx inspect i24-multiarch >/dev/null 2>&1 \
    || docker buildx create --name i24-multiarch --driver docker-container >/dev/null
  docker buildx build \
    --builder i24-multiarch \
    --platform linux/amd64,linux/arm64 \
    -f quick.Dockerfile \
    -t "${REPO}:latest" \
    --push \
    .
else
  docker build -f quick.Dockerfile -t i24-sim-quick:latest .
fi
