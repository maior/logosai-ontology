#!/bin/bash
# Ontology Builder Frontend (Next.js, port 9275)
#
# 기본은 빌드된 서버(`next start`)다 (2026-10-05). 개발 서버(`next dev`)로 상시 운영하자
# 감독의 3초 헬스체크가 거짓 '사망'을 52건 중 30건 냈다 — 잠자기에서 깨어날 때와 첫 컴파일
# (재기동 직후 첫 응답 4.6초 실측)에 응답이 늦다. 9275 는 관리 콘솔이라 HMR 이 상시 필요하지 않다.
#
#   ONTOLOGY_CONSOLE_MODE=dev  ./ontology/scripts/start_frontend.sh   # 콘솔을 고칠 때만
#
# 빌드가 없거나 소스가 빌드보다 새로우면 먼저 빌드한다 — 낡은 빌드를 조용히 띄우지 않는다.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")/frontend"
LOG_DIR="$(dirname "$SCRIPT_DIR")/logs"
PID_FILE="$LOG_DIR/ontology_frontend.pid"
BUILD_LOG="$LOG_DIR/frontend_build.log"
MODE="${ONTOLOGY_CONSOLE_MODE:-prod}"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "이미 실행 중 (PID $(cat "$PID_FILE"), port 9275)"
    exit 0
fi

cd "$FRONTEND_DIR"

if [ "$MODE" = "dev" ]; then
    NEXT_DIST_DIR=.next-dev nohup npm run dev >> "$LOG_DIR/ontology_frontend.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Ontology Builder Frontend 시작 (dev): http://localhost:9275 (PID $(cat "$PID_FILE"))"
    exit 0
fi

# ── prod: 빌드가 최신인지 확인 ────────────────────────────────────────
needs_build=false
if [ ! -f .next/BUILD_ID ]; then
    needs_build=true
elif [ -n "$(find app package.json package-lock.json next.config.mjs -newer .next/BUILD_ID -print -quit 2>/dev/null)" ]; then
    needs_build=true
fi

if $needs_build; then
    echo "빌드 중 (소스가 빌드보다 새롭거나 빌드 없음) — 로그: $BUILD_LOG"
    if ! npm run build > "$BUILD_LOG" 2>&1; then
        if [ -f .next/BUILD_ID ]; then
            echo "⚠️  빌드 실패 — 이전 빌드로 띄운다 (소스 변경이 반영되지 않았다). 로그: $BUILD_LOG"
        else
            echo "❌ 빌드 실패, 띄울 빌드가 없다. 로그: $BUILD_LOG" >&2
            exit 1
        fi
    fi
fi

nohup npm run start >> "$LOG_DIR/ontology_frontend.log" 2>&1 &
echo $! > "$PID_FILE"
echo "Ontology Builder Frontend 시작 (prod): http://localhost:9275 (PID $(cat "$PID_FILE"))"
