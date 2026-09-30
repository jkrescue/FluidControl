#!/usr/bin/env bash
set -euo pipefail

if docker info >/dev/null 2>&1; then
    echo "DOCKER_ALREADY_RUNNING"
    exit 0
fi

rm -f /var/run/docker.pid
nohup dockerd --host=unix:///var/run/docker.sock \
    >/tmp/dockerd-fluid-control.log 2>&1 </dev/null &

for _ in $(seq 1 30); do
    if docker info >/dev/null 2>&1; then
        echo "DOCKER_STARTED"
        docker version --format '{{.Server.Version}}'
        exit 0
    fi
    sleep 1
done

tail -n 80 /tmp/dockerd-fluid-control.log >&2
exit 1
