#!/bin/bash

set -e

DOC_STRING="Download the docker image for a quick evaluation of the system.

Pulls the multi-architecture i24-sim-quick image (SUMO only, no CARLA). Docker
selects the linux/amd64 or linux/arm64 variant matching this machine, so it runs
natively on both x86 Linux hosts and Apple Silicon Macs."

USAGE_STRING="Usage: $0 [--repo REPO] [-h|--help]

    --repo REPO    Registry repository to pull from (default: papapalpi/i24-sim-quick)"

REPO="papapalpi/i24-sim-quick"

# Plain argument loop: macOS ships BSD getopt, which has no --long support.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo ) REPO="$2"; shift 2 ;;
    -h | --help ) echo "$DOC_STRING"; echo; echo "$USAGE_STRING"; exit 0 ;;
    * ) echo "Unknown option: $1"; echo "$USAGE_STRING"; exit 1 ;;
  esac
done

echo "Pulling ${REPO}:latest"
docker pull "${REPO}:latest"
docker tag "${REPO}:latest" i24-sim-quick:latest
