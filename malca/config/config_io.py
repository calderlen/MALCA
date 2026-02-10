"""I/O settings: parquet compression, output formats, write chunk sizes."""

PARQUET_OUTPUT_COMPRESSION = "zstd"
PARQUET_CACHE_COMPRESSION = "snappy"
OUTPUT_FORMAT = "parquet"
EVENTS_OUTPUT_CHUNK_SIZE = 10000
INJECTION_CHUNK_SIZE = 1000
REPRODUCE_CHUNK_SIZE = 10000
