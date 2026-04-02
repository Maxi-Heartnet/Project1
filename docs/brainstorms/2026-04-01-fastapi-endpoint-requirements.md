---
date: 2026-04-01
topic: fastapi-endpoint
---

# API HTTP de Predicción de Precios

## Problem Frame

El chatbot CLI actual solo puede usarse de forma interactiva en la misma máquina. Para que otros puedan consultar predicciones en línea — desde una herramienta, un navegador, o un script — el modelo necesita exponerse como un endpoint HTTP accesible públicamente desde una instancia EC2 en AWS.

## Requirements

**API endpoint**
- R1. Un endpoint `POST /predict` acepta las características de una propiedad en JSON y devuelve el rango de precio estimado y el segmento de mercado.
- R2. El cuerpo de la request tiene los campos: `sector` (string), `property_type` (string: `apartment` o `house`), `bedrooms` (entero ≥ 1), `area_m2` (número > 0).
- R3. El cuerpo de la response incluye: `price_low`, `price_high`, `market_tier` (string), y `tier_stats` con los percentiles P10/P90 de precio, área y dormitorios para ese segmento.
- R4. Si los campos requeridos faltan o tienen valores inválidos, la API devuelve HTTP 422 con un mensaje descriptivo del error.
- R5. Un endpoint `GET /health` devuelve HTTP 200 para confirmar que el servidor está activo y el modelo cargado.

**Documentación automática**
- R6. La API expone documentación interactiva en `/docs` (Swagger UI) sin configuración adicional, donde cualquier persona puede probar el endpoint desde el navegador.

**Despliegue en EC2**
- R7. La API corre en una instancia EC2 Ubuntu (t3.small o equivalente) en el puerto 8000, accesible públicamente desde cualquier IP.
- R8. El proceso se mantiene activo después de cerrar la sesión SSH (mediante un process manager).
- R9. El modelo se carga una sola vez al arrancar el servidor, no en cada request.

**Acceso**
- R10. El endpoint es completamente público — sin autenticación requerida.
- R11. La API normaliza `sector` a título (`.title()`) y `property_type` a minúsculas (`.lower()`) antes de pasarlos al modelo, igual que el chatbot CLI.

## Success Criteria

- `curl -X POST http://<IP>:8000/predict -d '{...}'` devuelve una predicción válida desde cualquier red.
- `http://<IP>:8000/docs` muestra la documentación interactiva en el navegador.
- El servidor sigue corriendo después de cerrar y reabrir la terminal SSH.
- Una request con campos inválidos recibe HTTP 422, no un error 500.

## Scope Boundaries

- No hay autenticación — el endpoint es público.
- No hay HTTPS / dominio personalizado — solo IP pública con puerto 8000.
- No hay base de datos ni logging persistente de requests.
- No hay re-entrenamiento automático del modelo desde la API.
- No hay cambios al scraper, pipeline de datos, ni al chatbot CLI existente.
- No hay rate limiting en esta iteración.

## Key Decisions

- **FastAPI + Uvicorn**: Validación automática de inputs, documentación generada sin costo extra, y rendimiento suficiente para este volumen.
- **Modelo cargado al inicio**: Evita latencia por carga en cada request; el modelo cabe cómodamente en RAM de una t3.small.
- **Nueva función `predict_tier()` en `chatbot/chat.py`**: `display_tier()` ya existe y devuelve un string formateado para el CLI — se mantiene intacta. La API usará una nueva función `predict_tier()` que devuelve un dict estructurado `{market_tier, tier_stats}` con los mismos datos. Evita duplicar lógica y no rompe el chatbot existente.
- **Ruta absoluta para el modelo**: `MODEL_PATH` en `chatbot/chat.py` usa una ruta relativa que depende del directorio de trabajo del proceso. La API usará `Path(__file__).parent.parent / 'ml' / 'model.pkl'` para garantizar que funciona independientemente de dónde arranque Uvicorn o el process manager.
- **Normalización de inputs en la API**: `sector` se normaliza con `.title()` y `property_type` con `.lower()` — igual que el CLI — antes de pasar al modelo. Evita predicciones silenciosamente incorrectas por diferencias de capitalización.
- **Endpoint público**: Es una asignación académica / demo; la simplicidad tiene más valor que la seguridad en este contexto.

## Dependencies / Assumptions

- `ml/model.pkl` debe existir en la instancia EC2 antes de arrancar la API (se copia junto con el código o se re-entrena en la instancia).
- `fastapi` y `uvicorn[standard]` no están en `requirements.txt` aún — deben agregarse.
- La instancia EC2 tiene el puerto 8000 abierto en el Security Group hacia 0.0.0.0/0.
- El process manager (supervisor) debe configurar `directory=<raíz del proyecto>` para garantizar rutas relativas consistentes, aunque la API usará rutas absolutas como medida adicional.

## Outstanding Questions

### Deferred to Planning

- [Affects R8][Technical] ¿Supervisor o systemd para mantener el proceso activo? Ambos funcionan en Ubuntu 22.04; supervisor es más simple para un proyecto de aprendizaje.

## Next Steps
→ `/ce:plan` para planificación de implementación estructurada
