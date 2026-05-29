#!/bin/bash

set -e

DOC_STRING="Build docker images.

This script builds the base, development, and CI/CD docker images."

USAGE_STRING=$(cat <<- END
Usage: $0 [options]

Build configurations (choose one or more):

    --base               Build base docker image (required by dev, monolith and ci)
    --dev                Build development docker image
    --monolith           Build monolith docker image
    --ci                 Build CI/CD docker image

User and group options:

    --user UID:GID       Set host UID and GID for the container (default: current user)
    --docker-gid GID     Set GID of the Docker group (default: $(getent group docker | cut -d: -f3))

Ubuntu distribution:

    --ubuntu-distro DISTRO   Specify ubuntu distro (default: 20.04).

Build options:

    --force-rebuild      Force rebuild images with no cache
    --branch             CARLA branch (only for monolith configuration)

Epic credentials (only needed for monolith configuration)
    --epic-user          Github user name
    --epic-token         Github access token

Other commands:

    -h, --help           Show this help message and exit
END
)

UBUNTU_DISTRO=22.04

# CARLA target branch for monolith build
BRANCH="0.9.16_OpenTwinMap"
EPIC_USER=
EPIC_TOKEN=

HOST_UID=$(id -u)
HOST_GID=$(id -g)
DOCKER_GID=$(getent group docker | cut -d: -f3)

FORCE_REBUILD=

OPTS=`getopt -o h --long help,ubuntu-distro:,base,dev,monolith,ci,user:,docker-gid:,branch:,epic-user:,epic-token:,force-rebuild -n 'parse-options' -- "$@"`

eval set -- "$OPTS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ubuntu-distro )
      UBUNTU_DISTRO="$2";
      shift 2 ;;
    --user )
      IFS=':' read -r HOST_UID HOST_GID <<< "$2"
      shift 2 ;;
    --docker-gid)
      DOCKER_GID="$2"
      shift 2 ;;
    --branch)
      BRANCH="$2"
      shift 2 ;;
    --epic-user)
      EPIC_USER="$2"
      shift 2 ;;
    --epic-token)
      EPIC_TOKEN="$2"
      shift 2 ;;
    --force-rebuild )
      FORCE_REBUILD=true
      shift ;;
    -h | --help )
      echo "$DOC_STRING"
      echo "$USAGE_STRING"
      exit 1
      ;;
    * )
      shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARLA_ROOT="./i24motion_to_carla/carla-0.9.16"
I24MOTION_CARLA_ROOT="./i24motion_to_carla/i24motion_to_carla"
I24MOTION_MACRO_MICRO_ROOT="./i24motion_macro_micro/"

# Copy python runtime requirements for later installation in the docker image
rm -rf ${SCRIPT_DIR}/.tmp && mkdir ${SCRIPT_DIR}/.tmp
cp ${CARLA_ROOT}/PythonAPI/examples/requirements.txt ${SCRIPT_DIR}/.tmp/examples_requirements.txt
cp ${CARLA_ROOT}/PythonAPI/util/requirements.txt ${SCRIPT_DIR}/.tmp/util_requirements.txt
cp ${I24MOTION_CARLA_ROOT}/../requirements.txt ${SCRIPT_DIR}/.tmp/i24carla_requirements.txt
cp ${I24MOTION_MACRO_MICRO_ROOT}/requirements.txt ${SCRIPT_DIR}/.tmp/i24_macro_micro_requirements.txt

echo "Building base image carla-base:ue4-${UBUNTU_DISTRO}"

docker build \
  --build-arg UBUNTU_DISTRO=${UBUNTU_DISTRO} \
  -t carla-base:ue4-${UBUNTU_DISTRO} \
  -f ${CARLA_ROOT}/Util/Docker/Base.Dockerfile ${CARLA_ROOT}/Util/Docker/

if [ "$FORCE_REBUILD" = true ]; then
  echo "Removing existing volume i24-sim-${UBUNTU_DISTRO}"
  docker volume rm -f i24-sim-${UBUNTU_DISTRO} 2>/dev/null || true
fi
echo "Ensuring volume i24-sim-${UBUNTU_DISTRO} exists"
docker volume create i24-sim-volume-${UBUNTU_DISTRO}

echo "Building development image i24-sim-${UBUNTU_DISTRO} with user ${HOST_UID}:${HOST_GID}"
docker build ${FORCE_REBUILD:+--no-cache} \
  --build-arg UBUNTU_DISTRO=${UBUNTU_DISTRO} \
  --build-arg UID=${HOST_UID} \
  --build-arg GID=${HOST_GID} \
  --build-arg DOCKER_GID=${DOCKER_GID} \
  --target development \
  -t i24-sim-${UBUNTU_DISTRO} \
  -f ${SCRIPT_DIR}/container.Dockerfile ${SCRIPT_DIR}
