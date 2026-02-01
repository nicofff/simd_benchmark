# Rust SIMD Benchmark

Performance comparison of three SIMD implementations: Scalar vs std::simd vs NEON

**Tested on Apple M4 (MacBook Pro 2024)**

## Requirements

- Rust nightly (for `std::simd` / portable SIMD)
- Apple Silicon Mac (M1/M2/M3/M4) for NEON intrinsics

## Setup

```bash
rustup install nightly
rustup override set nightly  # Use nightly in current directory
```

## Running Benchmarks

### Quick Test

```bash
cargo run --release
```

### Criterion Detailed Benchmark

```bash
cargo bench
```

Results are saved in `target/criterion/` directory with HTML reports.

## Project Structure

```
src/
├── lib.rs          # Three implementations: scalar, portable_simd, neon
└── main.rs         # Simple benchmark runner
benches/
└── simd_bench.rs   # Criterion benchmarks
```

## Test Scenarios

| Category | Scenario | Description |
|----------|----------|-------------|
| Image Processing | RGB to Grayscale | 1920x1080 image conversion |
| Audio Processing | Volume Adjustment | 880K samples with gain control |
| Audio Processing | Two-track Mixing | Stereo audio mixing |
| String Search | Count Bytes | Count newlines in 10MB text |
| String Search | Find Byte | Find character position |
| Numerical | Dot Product | 10M element float vectors |
| Numerical | Matrix-Vector Multiply | 1024x1024 matrix |
| Validation | Range Check | Verify values in range |
| Validation | Sorted Check | Verify array is sorted |

## Blog Post

For detailed experimental analysis and code explanations, please see [BLOG.md](BLOG.md)
