#!/bin/bash

set -e

DOC_STRING="Run the quick-evaluation container (SUMO only, no CARLA).

Mounts i24motion_macro_micro at /workspaces/i24motion_macro_micro and publishes the
Dash viewer on port 8050. Works on x86_64 Linux and Apple Silicon macOS."

USAGE_STRING="Usage: $0 [--port PORT] [-h|--help]

    --port PORT    Host port for the Dash viewer (default: 8050)"

PORT=8050

# Plain argument loop: macOS ships BSD getopt, which has no --long support.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port ) PORT="$2"; shift 2 ;;
    -h | --help ) echo "$DOC_STRING"; echo; echo "$USAGE_STRING"; exit 0 ;;
    * ) echo "Unknown option: $1"; echo "$USAGE_STRING"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# -p rather than --net=host: host networking is off by default in Docker Desktop on
# macOS, so a server inside the container would not be reachable on localhost.
# Running as the host UID:GID keeps files written to run_data/ owned by you on Linux.
docker run \
  -it --rm \
  --name i24-sim-quick \
  --user "$(id -u):$(id -g)" \
  -p "${PORT}:8050" \
  -v "${SCRIPT_DIR}/i24motion_macro_micro:/workspaces/i24motion_macro_micro" \
  i24-sim-quick:latest bash
