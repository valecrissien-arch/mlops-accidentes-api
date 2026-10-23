# API FastAPI para predecir DIA_NOCHE en accidentes de tránsito
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# ---------------------------------------------------------
# CREDENCIALES DE DATABRICKS
# ---------------------------------------------------------
# Se leen desde variables de entorno.
# Nunca escribimos el token directamente en el código.
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
# ---------------------------------------------------------
# CREAR LA API
# ---------------------------------------------------------
app = FastAPI(
    title="API MLOps - Mortalidad Accidentes de Tránsito",
    description="Predicción de DIA_NOCHE usando el modelo desplegado en Databricks"
)
# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
# Permitirá que posteriormente nuestra página web
# pueda consumir esta API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
# ---------------------------------------------------------
# VARIABLES QUE RECIBE EL MODELO
# ---------------------------------------------------------
class Accidente(BaseModel):
    CICLO_VITAL: str
    CIRCUNSTANCIA_DEL_HECHO_DETALLADA: str
    CLASE_O_TIPO_DE_ACCIDENTE_DE_TRANSPORTE: str
    CONDICION_DE_LA_VICTIMA_AT_: str
    DIA_DEL_HECHO: str
    LOCALIDAD: str
    MEDIO_DE_DESPLAZAMIENTO_O_TRANSPORTE: str
    MES_DEL_HECHO: str
    OBJETO_DE_COLISION: str
    Sexo: str
# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "databricks_configurado": bool(
            DATABRICKS_URL and DATABRICKS_TOKEN
        )
    }
# ---------------------------------------------------------
# PREDICCION
# ---------------------------------------------------------
@app.post("/predecir")
def predecir(accidente: Accidente):
    # Verificamos que existan URL y token
    if not (DATABRICKS_URL and DATABRICKS_TOKEN):
        raise HTTPException(
            status_code=500,
            detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN"
        )
    # Convertimos los datos recibidos al formato
    # dataframe_records que acepta Databricks.
    payload = {
        "dataframe_records": [
            accidente.model_dump()
        ]
    }
    try:
        r = requests.post(
            DATABRICKS_URL,
            headers={
                "Authorization": f"Bearer {DATABRICKS_TOKEN}"
            },
            json=payload,
            timeout=120
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Databricks tardó demasiado en responder. Intente nuevamente."
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"No fue posible conectarse con Databricks: {str(e)}"
        )
    # Si Databricks devuelve un error
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Error de Databricks: {r.text}"
        )
    # Leer la respuesta de Databricks
    respuesta = r.json()
    if "predictions" not in respuesta:
        raise HTTPException(
            status_code=502,
            detail=f"Databricks no devolvió predictions: {respuesta}"
        )
    prediccion = respuesta["predictions"][0]
    # -----------------------------------------------------
    # RESPUESTA DE NUESTRA API
    # -----------------------------------------------------
    return {
        "prediccion": prediccion,
        "resultado": str(prediccion),
        "datos_recibidos": accidente.model_dump()
    }