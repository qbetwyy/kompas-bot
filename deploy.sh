#!/usr/bin/env bash
# Установка «Компаса» на сервер (Ubuntu/Debian), запускать от root:
#   bash <(curl -fsSL https://raw.githubusercontent.com/qbetwyy/kompas-bot/main/deploy.sh)
# Скрипт спросит токен бота и ключ Anthropic (ввод скрыт) и запустит бота как службу systemd.
# Другие боты на сервере не трогаются: всё ставится в /opt/kompas_bot, служба называется kompas.
set -euo pipefail
APP=/opt/kompas_bot
REPO=https://github.com/qbetwyy/kompas-bot.git

if [ "$(id -u)" != "0" ]; then echo "Запусти от root (или через sudo)"; exit 1; fi

echo "==> Ставлю системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y -qq
apt-get install -y -qq python3-venv python3-pip git curl fonts-dejavu-core >/dev/null

PYV=$(python3 -c 'import sys; print(sys.version_info >= (3, 10))')
if [ "$PYV" != "True" ]; then
  echo "Нужен Python 3.10 или новее, а тут $(python3 -V). Поставь Ubuntu 22.04/24.04 или python3.11."
  exit 1
fi

echo "==> Забираю код"
if [ -d "$APP/.git" ]; then git -C "$APP" pull -q --ff-only; else git clone -q "$REPO" "$APP"; fi
cd "$APP"

# Сколько памяти: на маленьком сервере берём лёгкую модель распознавания речи
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
if [ "$MEM_MB" -lt 1200 ]; then WHISPER=tiny; else WHISPER=base; fi
echo "    памяти ${MEM_MB} МБ, модель для голоса: ${WHISPER}"

echo "==> Ставлю зависимости (пара минут)"
python3 -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt faster-whisper

if [ ! -f .env ]; then
  cp .env.example .env
  T=""; K=""
  while [ -z "$T" ]; do read -rsp "Вставь токен бота (BOT_TOKEN) и нажми Enter: " T; echo; T=$(printf %s "$T" | tr -d '[:space:]'); T=${T#BOT_TOKEN=}; done
  while [ -z "$K" ]; do read -rsp "Вставь ключ Anthropic (ANTHROPIC_API_KEY) и нажми Enter: " K; echo; K=$(printf %s "$K" | tr -d '[:space:]'); K=${K#ANTHROPIC_API_KEY=}; done
  sed -i "s|^BOT_TOKEN=.*|BOT_TOKEN=${T}|; s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${K}|; s|^WHISPER_MODEL=.*|WHISPER_MODEL=${WHISPER}|" .env
  chmod 600 .env
  unset T K
fi

echo "==> Скачиваю модель для голосовых (один раз)"
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER}', device='cpu', compute_type='int8')" >/dev/null 2>&1 \
  && echo "    готово" || echo "    не вышло, скачается при первом голосовом"

echo "==> Запускаю службу kompas"
cat >/etc/systemd/system/kompas.service <<UNIT
[Unit]
Description=Kompas Telegram bot
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=$APP
ExecStart=$APP/.venv/bin/python bot.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable -q kompas
systemctl restart kompas
sleep 10
journalctl -u kompas -n 15 --no-pager
if journalctl -u kompas -n 50 --no-pager | grep -qi "conflict"; then
  echo "!!! Бот запущен ещё где-то (скорее всего на маке). Останови его там, иначе они мешают друг другу."
fi
echo "==> Готово. Логи: journalctl -u kompas -f   Перезапуск: systemctl restart kompas"
