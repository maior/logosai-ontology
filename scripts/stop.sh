#!/bin/bash
# Ontology Builder Server 중지
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$(dirname "$SCRIPT_DIR")/logs/ontology_server.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
    echo "Ontology Builder Server 중지됨"
else
    # PID 파일이 없으면 포트 기준으로 정리
    PID=$(lsof -ti :9274 2>/dev/null)
    if [ -n "$PID" ]; then
        kill "$PID"
        echo "Ontology Builder Server 중지됨 (port 기준, PID $PID)"
    else
        echo "실행 중인 서버 없음"
    fi
    rm -f "$PID_FILE"
fi
