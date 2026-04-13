; Supervisor configuration for the Santo Domingo House Price API
;
; This is a Terraform templatefile — rendered by `local_file.supervisor_conf` in main.tf.
; Do NOT copy this file directly to /etc/supervisor/conf.d/. Use the rendered output.
;
; NOTES:
;   - Update 'command' path if your virtualenv is named differently.
;   - Update 'directory' if the project is cloned to a different path.
;   - stopasgroup + killasgroup are required to also kill Uvicorn worker
;     child processes on restart; without them, workers become orphans.

[program:predict-api]

; Full path to uvicorn inside the virtualenv (avoids PATH issues)
command=/home/ubuntu/Project1/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2

; Project root — must match the directory where api.py lives so that
; 'from chatbot.chat import ...' resolves correctly
directory=/home/ubuntu/Project1

user=ubuntu

; Environment variables injected at deploy time via Terraform templatefile().
; $${...} is Terraform interpolation syntax, not shell.
; If supervisor fails to start after adding new env vars, check:
;   /var/log/predict-api/stderr.log
environment=GOOGLE_MAPS_API_KEY="${google_maps_api_key}",MAPS_MAP_ID="${maps_map_id}"

; Start automatically when supervisord starts (survives reboots)
autostart=true

; Restart the process if it exits unexpectedly
autorestart=true

; Kill all worker child processes cleanly on stop/restart
stopasgroup=true
killasgroup=true

; Logs — check these if the API fails to start
stdout_logfile=/var/log/predict-api/stdout.log
stderr_logfile=/var/log/predict-api/stderr.log
stdout_logfile_maxbytes=10MB
stderr_logfile_maxbytes=10MB
