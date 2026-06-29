from pathlib import Path
import pandas as pd
import pyreadstat


PROJECT = Path(r"F:\MU\Research\ML\NHANES_Project")

RAW = PROJECT / "data" / "raw"
INTERIM = PROJECT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

DATA_FOLDERS = ["demographics", "examination", "laboratory", "questionnaire"]

CYCLE_MAP = {
    "G": "2011-2012",
    "H": "2013-2014",
    "I": "2015-2016",
    "J": "2017-2018",
}


def read_xpt(file_path):
    df, meta = pyreadstat.read_xport(str(file_path))
    return df


def get_cycle_from_filename(file_path):
    suffix = file_path.stem.split("_")[-1]
    return CYCLE_MAP.get(suffix)


def merge_cycle(files):
    merged = None

    for file_path in files:
        df = read_xpt(file_path)

        if "SEQN" not in df.columns:
            print(f"Skipped no SEQN: {file_path.name}")
            continue

        print(f"Loaded {file_path.name}: {df.shape}")

        if merged is None:
            merged = df
        else:
            duplicate_cols = [
                c for c in df.columns
                if c in merged.columns and c != "SEQN"
            ]
            df = df.drop(columns=duplicate_cols, errors="ignore")
            merged = merged.merge(df, on="SEQN", how="left")

    return merged


def main():
    all_files = []

    for folder in DATA_FOLDERS:
        folder_path = RAW / folder
        all_files.extend(folder_path.glob("*.XPT"))

    print(f"Found {len(all_files)} XPT files")

    cycle_files = {}

    for file_path in all_files:
        cycle = get_cycle_from_filename(file_path)
        if cycle is None:
            print(f"Unknown cycle skipped: {file_path.name}")
            continue

        cycle_files.setdefault(cycle, []).append(file_path)

    merged_cycles = []

    for cycle, files in cycle_files.items():
        print("\n" + "=" * 50)
        print(f"Merging cycle: {cycle}")
        print("=" * 50)

        merged = merge_cycle(files)

        if merged is None:
            continue

        merged["cycle"] = cycle

        output_file = INTERIM / f"nhanes_{cycle}_merged.csv"
        merged.to_csv(output_file, index=False)

        print(f"Saved: {output_file}")
        print(f"Shape: {merged.shape}")

        merged_cycles.append(merged)

    master = pd.concat(merged_cycles, ignore_index=True)

    master_file = INTERIM / "nhanes_master_2011_2014.csv"
    master.to_csv(master_file, index=False)

    print("\nDONE")
    print(f"Master file saved: {master_file}")
    print(f"Master shape: {master.shape}")


if __name__ == "__main__":
    main()