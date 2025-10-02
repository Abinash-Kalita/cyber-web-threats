import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop duplicates
    df.drop_duplicates(inplace=True)

    # 2. Convert datetime columns safely
    for col in ["creation_time", "end_time", "time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 3. Feature engineering
    if "end_time" in df.columns and "creation_time" in df.columns:
        df["session_duration"] = (df["end_time"] - df["creation_time"]).dt.total_seconds().fillna(0)
    else:
        df["session_duration"] = 0  # fallback

    df["bytes_total"] = df.get("bytes_in", 0).fillna(0) + df.get("bytes_out", 0).fillna(0)
    df["avg_packet_size"] = df["bytes_total"] / df["session_duration"].replace(0, 1)

    # 4. Standardize country code if present
    if "src_ip_country_code" in df.columns:
        df["src_ip_country_code"] = df["src_ip_country_code"].astype(str).str.upper()

    return df


if __name__ == "__main__":
    # quick test
    from load_data import load_data

    df = load_data()
    df_clean = preprocess(df)

    print("✅ Preprocessing complete")
    print("Shape:", df_clean.shape)
    print("New columns:", [c for c in df_clean.columns if c in ["session_duration","bytes_total","avg_packet_size"]])
    print(df_clean.head())
