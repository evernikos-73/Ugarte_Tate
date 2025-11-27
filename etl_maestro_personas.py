# etl_maestro_personas.py

import os
import sys
import io
import json

import numpy as np
import pandas as pd
import requests
import sqlalchemy as sa
from urllib.parse import quote_plus


# ---------------------------------------------------------
# Conexión a SQL Server
# ---------------------------------------------------------
def get_engine():
    host = os.environ["SQLSERVER_HOST"]
    db = os.environ["SQLSERVER_DB"]
    user = os.environ["SQLSERVER_USER"]
    password = os.environ["SQLSERVER_PASSWORD"]
    driver = os.environ.get("SQLSERVER_DRIVER", "ODBC Driver 18 for SQL Server")

    odbc_str = (
        f"DRIVER={driver};"
        f"SERVER={host};"
        f"DATABASE={db};"
        f"UID={user};PWD={password};"
        "Encrypt=no;TrustServerCertificate=yes;"
    )

    connect_str = "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_str)
    engine = sa.create_engine(connect_str, fast_executemany=True)
    return engine


# ---------------------------------------------------------
# Leer CSV enriquecido desde Google Drive
# ENRICHED_CSV_URL = link de descarga directa (uc?export=download&id=...)
# ---------------------------------------------------------
def download_enriched_csv() -> pd.DataFrame:
    url = os.environ["ENRICHED_CSV_URL"]
    resp = requests.get(url)
    resp.raise_for_status()

    df = pd.read_csv(io.BytesIO(resp.content), dtype=str)

    # Renombramos columnas a los nombres usados en el script
    df = df.rename(
        columns={
            "Calle": "Calle_Enr",
            "Nombre": "Nombre_Enr",
            "Apellido": "Apellido_Enr",
            "Anio_Nac": "Anio_Nac_Enr",
            "Provincia": "Provincia_Enr",
            "Localidad": "Localidad_Enr",
            "Genero": "Genero_Enr",
        }
    )

    df["DNI"] = pd.to_numeric(df["DNI"], errors="coerce")
    df["Anio_Nac_Enr"] = pd.to_numeric(df["Anio_Nac_Enr"], errors="coerce")

    return df


# ---------------------------------------------------------
# Parseo del JSON ROL_Json (equivalente a Json.Document + expand)
# ---------------------------------------------------------
def parse_rol_json_column(df: pd.DataFrame) -> pd.DataFrame:
    # Default para nulos
    df["ROL_Json_Clean"] = df["ROL_Json"].fillna('{"resultado":[ ]}')

    # Manejo de error 502: si contiene ese texto, reemplazamos TODO por el stub
    mask_502 = df["ROL_Json_Clean"].str.contains("error code: 502", na=False)
    df.loc[mask_502, "ROL_Json_Clean"] = (
        '{"resultado":[{"cuit":null,"nombre":"ERROR 502",'
        '"documento":null,"domicilio":"N/A","actividad":"N/A"}]}'
    )

    def parse_cell(s):
        try:
            data = json.loads(s)
            res = data.get("resultado") or []
            if not res:
                return [None, "", None, None, None, None]
            item = res[0] or {}
            return [
                item.get("cuit"),
                item.get("nombre") or "",
                item.get("sexo"),
                item.get("clase"),
                item.get("domicilio"),
                item.get("actividad"),
            ]
        except Exception:
            return [None, "", None, None, None, None]

    parsed = df["ROL_Json_Clean"].apply(parse_cell).tolist()
    cols = [
        "CuitJson",
        "NombreJson",
        "SexoJson",
        "ClaseJson",
        "DomicilioJsonFull",
        "ActividadJson",
    ]
    parsed_df = pd.DataFrame(parsed, columns=cols, index=df.index)
    df = pd.concat([df, parsed_df], axis=1)

    return df


# ---------------------------------------------------------
# Match Nombres – versión Jaccard como en tu M
# ---------------------------------------------------------
def compute_match_nombres_jaccard(df: pd.DataFrame) -> pd.DataFrame:
    texto1 = (
        df["Nombre"].fillna("")
        .str.cat(df["Apellido"].fillna(""), sep=" ")
        .str.strip()
        .str.lower()
    )
    texto2 = df["NombreJson"].fillna("").str.strip().str.lower()

    def jaccard(s1, s2):
        if not s1:
            lista1 = []
        else:
            lista1 = s1.split()

        if not s2:
            lista2 = []
        else:
            lista2 = s2.split()

        set1, set2 = set(lista1), set(lista2)
        if not set1 and not set2:
            return 0.0
        inter = len(set1 & set2)
        union = len(set1 | set2)
        return inter / union if union else 0.0

    df["Match_Nombres_Score"] = [
        jaccard(a, b) for a, b in zip(texto1.tolist(), texto2.tolist())
    ]
    return df


# ---------------------------------------------------------
# Parseo de domicilio tipo "CALLE, LOCALIDAD, PROVINCIA"
# (equivalente al doble SplitColumn + lógica N Prov / N Loc / N Dom)
# ---------------------------------------------------------
def parse_address(addr: str):
    if not isinstance(addr, str):
        return (None, None, None)
    addr = addr.strip()
    if not addr:
        return (None, None, None)
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    if not parts:
        return (None, None, None)
    # Primera parte = domicilio, segunda = localidad, tercera = provincia
    domicilio = parts[0]
    localidad = parts[1] if len(parts) >= 2 else None
    provincia = parts[2] if len(parts) >= 3 else None
    return (provincia, localidad, domicilio)


# ---------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------
def main():
    engine = get_engine()

    # 1) Leemos Maestro_Personas de SQL (equivalente a Origen en M)
    base_query = """
        SELECT 
            p.Idpersona,
            TRY_CAST(p.DNI AS BIGINT) AS DNI,
            p.FechaNac,
            p.Nombre,
            p.Apellido,
            p.Emails,
            p.Celulares,
            p.Fuentes,
            p.Controlado,
            p.IdTrii,
            p.Rol,
            p.Rol_F_Control,
            p.Fecha_Control_Pagina,
            p.idusuario_control,
            p.ROL_Json
        FROM dbo.Maestro_Personas p;
    """
    df = pd.read_sql(base_query, engine)

    df["DNI"] = pd.to_numeric(df["DNI"], errors="coerce")

    # 2) Parseo JSON + limpieza error 502
    df = parse_rol_json_column(df)

    # 3) Match Nombres (Jaccard)
    df = compute_match_nombres_jaccard(df)

    # 4) Columna "DNI fuera rango" (no la usamos al final, pero la calculamos)
    df["DNI_fuera_rango"] = np.where(
        (df["DNI"] < 5_000_000) | (df["DNI"] > 100_000_000),
        "DNI fuera de rango",
        "ok",
    )

    # 5) Join con Maestro_Personas_Enriquecido (CSV en Drive)
    df_enr = download_enriched_csv()

    df = df.merge(df_enr, on="DNI", how="left")

    # 6) Columna "Check Coincidencia Genero" (sexo = Genero)
    df["Check_Coincidencia_Genero"] = (
        df["SexoJson"].fillna("") == df["Genero_Enr"].fillna("")
    )

    # 7) Join con Vista_DNI_Duplicados
    df_dup = pd.read_sql(
        "SELECT DNI, Duplicado FROM dbo.Vista_DNI_Duplicados", engine
    )
    df_dup["DNI"] = pd.to_numeric(df_dup["DNI"], errors="coerce")

    df = df.merge(df_dup, on="DNI", how="left")

    # 8) IndexOriginal (equivalente al índice agregado en M)
    df["IndexOriginal"] = np.arange(len(df), dtype="int64")

    # 9) MarcaTemporal (igual que en M)
    df["MarcaTemporal"] = np.where(
        (df["Duplicado"] == "Duplicado")
        & (~df["Check_Coincidencia_Genero"])
        & (df["Match_Nombres_Score"] < 0.2),
        "Eliminar",
        "",
    )

    # 10) Lógica MarcaFinal por DNI (superviviente por DNI)
    #     – ordenamos por DNI, MarcaTemporal (bueno primero), IndexOriginal
    df["MarcaScore"] = np.where(df["MarcaTemporal"] == "", 0, 1)
    df_sorted = df.sort_values(["DNI", "MarcaScore", "IndexOriginal"])

    # groupby con dropna=False para incluir DNI nulos (como M)
    survivors = (
        df_sorted.groupby("DNI", dropna=False, as_index=False).head(1).copy()
    )

    # 11) Parsing de domicilio como en M
    addr_parts = survivors["DomicilioJsonFull"].apply(parse_address).tolist()
    addr_df = pd.DataFrame(
        addr_parts, columns=["ProvinciaJson", "LocalidadJson", "DomicilioJson"]
    )
    survivors = pd.concat([survivors, addr_df], axis=1)

    # N Prov, N Loc, N Dom
    survivors["ProvinciaFinal"] = survivors["ProvinciaJson"].where(
        survivors["ProvinciaJson"].notna(), survivors["Provincia_Enr"]
    )
    survivors["LocalidadFinal"] = survivors["LocalidadJson"].where(
        survivors["LocalidadJson"].notna(), survivors["Localidad_Enr"]
    )
    dom_tmp = survivors["DomicilioJson"].where(
        survivors["DomicilioJson"].notna(), survivors["Calle_Enr"]
    )
    survivors["DireccionFinal"] = dom_tmp.where(
        dom_tmp.notna(), survivors["Calle_Enr"]
    )

    # Proper case (Text.Proper)
    survivors["ProvinciaFinal"] = (
        survivors["ProvinciaFinal"].fillna("").str.title()
    )
    survivors["LocalidadFinal"] = (
        survivors["LocalidadFinal"].fillna("").str.title()
    )
    survivors["DireccionFinal"] = (
        survivors["DireccionFinal"].fillna("").str.title()
    )

    # 12) Año Nac (mismo criterio que M)
    # if FechaNac = null -> Anio_Nac_Enr
    # else if ClaseJson = null -> Anio_Nac_Enr
    # else ClaseJson
    clase_int = pd.to_numeric(survivors["ClaseJson"], errors="coerce")
    mask_use_clase = survivors["FechaNac"].notna() & clase_int.notna()
    anio_nac_final = np.where(
        mask_use_clase, clase_int, survivors["Anio_Nac_Enr"]
    )
    survivors["Anio_Nac_Final"] = pd.to_numeric(
        anio_nac_final, errors="coerce"
    ).astype("Int64")

    # 13) Género final (New Gender en M)
    genero_raw = survivors["Genero_Enr"].fillna("").astype(str)
    sexo_raw = survivors["SexoJson"].fillna("").astype(str)

    mask_genero_indef = genero_raw.str.lower().isin(
        ["", "unknown", "a"]
    )
    genero_tmp = np.where(
        mask_genero_indef,
        np.where(sexo_raw.replace("nan", "") == "", genero_raw, sexo_raw),
        genero_raw,
    )
    genero_final = pd.Series(genero_tmp).str.upper().str[:1]
    survivors["GeneroFinal"] = genero_final

    # 14) Armamos DataFrame final (zMaestro_Personas_PowerBI)
    #     – aquí defino columnas "útiles" para Power BI; puedes sumar más si las necesitas
    df_out = survivors[
        [
            "Idpersona",
            "DNI",
            "Nombre",
            "Apellido",
            "Emails",
            "Celulares",
            "Fuentes",
            "Segmento",
            "GeneroFinal",
            "ProvinciaFinal",
            "LocalidadFinal",
            "DireccionFinal",
            "Anio_Nac_Final",
            "Match_Nombres_Score",
            "MarcaTemporal",
        ]
    ].rename(
        columns={
            "GeneroFinal": "Genero",
            "ProvinciaFinal": "Provincia",
            "LocalidadFinal": "Localidad",
            "DireccionFinal": "Direccion",
            "Anio_Nac_Final": "Anio_Nac",
        }
    )

    # 15) Crear / truncar tabla destino zMaestro_Personas_PowerBI
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
        IF OBJECT_ID('dbo.zMaestro_Personas_PowerBI','U') IS NULL
        BEGIN
            CREATE TABLE dbo.zMaestro_Personas_PowerBI (
                Idpersona BIGINT NOT NULL,
                DNI BIGINT NULL,
                Nombre NVARCHAR(255) NULL,
                Apellido NVARCHAR(255) NULL,
                Emails NVARCHAR(4000) NULL,
                Celulares NVARCHAR(4000) NULL,
                Fuentes NVARCHAR(4000) NULL,
                Segmento NVARCHAR(255) NULL,
                Genero NCHAR(1) NULL,
                Provincia NVARCHAR(255) NULL,
                Localidad NVARCHAR(255) NULL,
                Direccion NVARCHAR(1000) NULL,
                Anio_Nac INT NULL,
                Match_Nombres_Score FLOAT NULL,
                MarcaTemporal NVARCHAR(50) NULL
            );
        END;
        TRUNCATE TABLE dbo.zMaestro_Personas_PowerBI;
        """
            )
        )

    # 16) Insert masivo en zMaestro_Personas_PowerBI
    df_out.to_sql(
        "zMaestro_Personas_PowerBI",
        engine,
        schema="dbo",
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi",
    )

    # 17) Índice clustered en Idpersona (si no existe)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
        IF NOT EXISTS (
            SELECT 1 FROM sys.indexes
            WHERE name = 'IX_zMaestro_Personas_PowerBI_Idpersona'
              AND object_id = OBJECT_ID('dbo.zMaestro_Personas_PowerBI')
        )
        BEGIN
            CREATE CLUSTERED INDEX IX_zMaestro_Personas_PowerBI_Idpersona
            ON dbo.zMaestro_Personas_PowerBI (Idpersona);
        END;
        """
            )
        )

    print(f"Proceso terminado. Filas finales: {len(df_out)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error en ETL:", e, file=sys.stderr)
        sys.exit(1)
