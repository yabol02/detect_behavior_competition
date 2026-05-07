import polars as pl


def load_ts_data(path: str, null_values: list[str]) -> pl.LazyFrame:
    lf = pl.scan_csv(path, infer_schema=True, null_values=null_values)
    # .sort(["sequence_id", "sequence_counter"])
    # .drop(pl.col("row_id"))
    # .with_columns(
    #     pl.col(pl.Float64).cast(pl.Float32),
    #     pl.col(
    #         [
    #             "sequence_type",
    #             "sequence_id",
    #             "subject",
    #             "orientation",
    #             "phase",
    #             "gesture",
    #         ]
    #     ).cast(pl.Categorical),
    # )
    schema = lf.collect_schema()
    existing_cols = set(schema.names())

    categorical_candidates = [
        "sequence_type",
        "sequence_id",
        "subject",
        "orientation",
        "phase",
        "gesture",
        # "row_id",  # <== Solo aparece en test (???)
    ]

    categorical_cols = [c for c in categorical_candidates if c in existing_cols]

    lf = (
        lf.sort(["sequence_id", "sequence_counter"])
        .drop([c for c in ["row_id"] if c in existing_cols])
        .with_columns(
            pl.col(pl.Float64).cast(pl.Float32),
            pl.col(categorical_cols).cast(pl.Categorical) if categorical_cols else [],
        )
    )

    return lf


def preprocess_pipeline(lf: pl.LazyFrame) -> pl.DataFrame:
    """Realiza el pipeline completo de preprocesamiento:
    - Validación de secuencias (asegura que sequence_counter sea creciente dentro de cada sequence_id).
    - Interpolación de valores faltantes en columnas Float32 escalares.
    - Normalización por sujeto (media 0, desviación estándar 1).
    - Adición de características temporales (diferencias, medias móviles, magnitud de aceleración).

    :param lf: LazyFrame de entrada con columnas originales.
    :return: DataFrame preprocesado listo para análisis o modelado.
    """
    return (
        lf.pipe(group_tof_sensors)
        .pipe(validate_sequences)
        .pipe(interpolate_missing)
        .pipe(normalize_per_subject)
        .pipe(add_temporal_features)
        .collect()
    )


def group_tof_sensors(lf: pl.LazyFrame) -> pl.LazyFrame:
    for i in range(1, 6):
        tof_cols = [f"tof_{i}_v{j}" for j in range(64)]
        lf = lf.with_columns(pl.concat_arr(tof_cols).alias(f"tof_{i}"))
    return lf.drop(pl.selectors.matches(r"^tof_\d_v\d+$"))


def _scalar_float_cols(lf: pl.LazyFrame) -> list[str]:
    """Identifica las columnas de tipo Float32 que son escalares (excluye las columnas de tipo Array como tof_* y sequence_counter).

    :param lf: LazyFrame de entrada.
    :return: Lista de nombres de columnas de tipo Float32 que son escalares.
    """
    return [name for name, dtype in lf.collect_schema().items() if dtype == pl.Float32]


def validate_sequences(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Valida que dentro de cada sequence_id, el sequence_counter sea estrictamente creciente.
    Filtra las filas que no cumplen esta condición.

    :param lf: LazyFrame de entrada.
    :return: LazyFrame con las filas válidas.
    """
    return (
        lf.with_columns(
            is_valid_step=(
                pl.col("sequence_counter").diff().over("sequence_id") > 0
            ).fill_null(True)
        )
        .filter(pl.col("is_valid_step"))
        .drop("is_valid_step")
    )


def interpolate_missing(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Realiza interpolación lineal para las columnas de tipo Float32 escalares, llenando los valores faltantes dentro de cada sequence_id.
    Utiliza interpolación lineal, seguido de forward fill y backward fill para asegurar que no queden valores nulos.

    :param lf: LazyFrame de entrada.
    :return: LazyFrame con las columnas de tipo Float32 escalares interpoladas y sin valores faltantes.
    """
    float_cols = _scalar_float_cols(lf)
    return lf.with_columns(
        [
            pl.col(c).interpolate().forward_fill().backward_fill().over("sequence_id")
            for c in float_cols
        ]
    )


def normalize_per_subject(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Realiza normalización por sujeto para las columnas de tipo Float32 escalares, calculando la media y desviación estándar por sujeto y aplicando la fórmula de normalización (valor - media) / desviación estándar.
    Agrega temporalmente las columnas de media y desviación estándar para cada columna, luego las utiliza para normalizar y finalmente las elimina del DataFrame resultante.

    :param lf: LazyFrame de entrada.
    :return: LazyFrame con las columnas de tipo Float32 escalares normalizadas.
    """
    float_cols = _scalar_float_cols(lf)

    stats = lf.group_by("subject").agg(
        [pl.col(c).mean().alias(f"{c}_mean") for c in float_cols]
        + [pl.col(c).std().alias(f"{c}_std") for c in float_cols]
    )

    lf = lf.join(stats, on="subject", how="left")

    lf = lf.with_columns(
        [
            ((pl.col(c) - pl.col(f"{c}_mean")) / (pl.col(f"{c}_std") + 1e-6)).alias(c)
            for c in float_cols
        ]
    )

    drop_cols = [f"{c}_mean" for c in float_cols] + [f"{c}_std" for c in float_cols]
    return lf.drop(drop_cols)


def add_temporal_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Agrega características temporales a las señales IMU:
    - Diferencias entre pasos consecutivos para cada señal (diff).
    - Medias móviles con ventana de 5 pasos para cada señal (rolling_mean).
    - Magnitud de la aceleración combinando acc_x, acc_y y acc_z.

    :param lf: LazyFrame de entrada.
    :return: LazyFrame con las características temporales agregadas.
    """
    imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
    acc_cols = ["acc_x", "acc_y", "acc_z"]

    diff_exprs = [
        pl.col(c).diff().over("sequence_id").fill_null(0).alias(f"{c}_diff")
        for c in imu_cols
    ]
    roll_mean_exprs = [
        pl.col(c)
        .rolling_mean(window_size=5, min_samples=1)
        .over("sequence_id")
        .alias(f"{c}_rmean5")
        for c in imu_cols
    ]

    acc_magnitude = (
        (pl.col("acc_x") ** 2 + pl.col("acc_y") ** 2 + pl.col("acc_z") ** 2)
        .sqrt()
        .alias("acc_magnitude")
    )

    return lf.with_columns(diff_exprs + roll_mean_exprs + [acc_magnitude])
