#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use simd_benchmark::{image_processing, audio_processing, string_search, numerical, validation};

const SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

// =============================================================================
// Image Processing Benchmarks
// =============================================================================

fn bench_rgb_to_grayscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("RGB to Grayscale");

    // Use smaller sizes for image (each pixel = 3 bytes)
    let image_sizes: &[usize] = &[10_000, 100_000, 1_000_000, 2_073_600]; // last = 1920x1080

    for &size in image_sizes {
        let rgb: Vec<u8> = (0..size * 3).map(|i| (i % 256) as u8).collect();
        let mut gray = vec![0u8; size];

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, _| {
            bencher.iter(|| image_processing::scalar::rgb_to_grayscale(black_box(&rgb), black_box(&mut gray)));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", size), &size, |bencher, _| {
            bencher.iter(|| image_processing::portable_simd::rgb_to_grayscale(black_box(&rgb), black_box(&mut gray)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", size), &size, |bencher, _| {
            bencher.iter(|| image_processing::neon::rgb_to_grayscale(black_box(&rgb), black_box(&mut gray)));
        });
    }

    group.finish();
}

// =============================================================================
// Audio Processing Benchmarks
// =============================================================================

fn bench_audio_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("Audio Volume Adjustment");

    for &size in SIZES {
        let mut samples: Vec<i16> = (0..size).map(|i| (i as i16).wrapping_mul(7)).collect();
        let samples_clone = samples.clone();

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, _| {
            samples.copy_from_slice(&samples_clone);
            bencher.iter(|| audio_processing::scalar::adjust_volume(black_box(&mut samples), black_box(0.8)));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", size), &size, |bencher, _| {
            samples.copy_from_slice(&samples_clone);
            bencher.iter(|| audio_processing::portable_simd::adjust_volume(black_box(&mut samples), black_box(0.8)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", size), &size, |bencher, _| {
            samples.copy_from_slice(&samples_clone);
            bencher.iter(|| audio_processing::neon::adjust_volume(black_box(&mut samples), black_box(0.8)));
        });
    }

    group.finish();
}

fn bench_audio_mix(c: &mut Criterion) {
    let mut group = c.benchmark_group("Audio Mixing");

    for &size in SIZES {
        let track1: Vec<i16> = (0..size).map(|i| (i % 32768) as i16).collect();
        let track2: Vec<i16> = (0..size).map(|i| ((i % 32768) as i16).wrapping_neg()).collect();
        let mut output = vec![0i16; size];

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, _| {
            bencher.iter(|| audio_processing::scalar::mix_tracks(black_box(&track1), black_box(&track2), black_box(&mut output)));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", size), &size, |bencher, _| {
            bencher.iter(|| audio_processing::portable_simd::mix_tracks(black_box(&track1), black_box(&track2), black_box(&mut output)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", size), &size, |bencher, _| {
            bencher.iter(|| audio_processing::neon::mix_tracks(black_box(&track1), black_box(&track2), black_box(&mut output)));
        });
    }

    group.finish();
}

// =============================================================================
// String Search Benchmarks
// =============================================================================

fn bench_count_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("Count Byte (Newlines)");

    for &size in SIZES {
        // Create text with newlines every ~80 chars
        let data: Vec<u8> = (0..size).map(|i| if i % 80 == 0 { b'\n' } else { b'x' }).collect();

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, _| {
            bencher.iter(|| string_search::scalar::count_byte(black_box(&data), black_box(b'\n')));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", size), &size, |bencher, _| {
            bencher.iter(|| string_search::portable_simd::count_byte(black_box(&data), black_box(b'\n')));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", size), &size, |bencher, _| {
            bencher.iter(|| string_search::neon::count_byte(black_box(&data), black_box(b'\n')));
        });
    }

    group.finish();
}

fn bench_find_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("Find Byte Position");

    for &size in SIZES {
        // Put target byte near the end
        let mut data: Vec<u8> = vec![b'x'; size];
        if size > 100 {
            data[size - 100] = b'@';
        }

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, _| {
            bencher.iter(|| string_search::scalar::find_byte(black_box(&data), black_box(b'@')));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", size), &size, |bencher, _| {
            bencher.iter(|| string_search::portable_simd::find_byte(black_box(&data), black_box(b'@')));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", size), &size, |bencher, _| {
            bencher.iter(|| string_search::neon::find_byte(black_box(&data), black_box(b'@')));
        });
    }

    group.finish();
}

// =============================================================================
// Numerical Computing Benchmarks
// =============================================================================

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dot Product");

    for &size in SIZES {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.0001).collect();
        let b: Vec<f32> = (0..size).map(|i| ((size - i) as f32) * 0.0001).collect();

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, _| {
            bencher.iter(|| numerical::scalar::dot_product(black_box(&a), black_box(&b)));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", size), &size, |bencher, _| {
            bencher.iter(|| numerical::portable_simd::dot_product(black_box(&a), black_box(&b)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", size), &size, |bencher, _| {
            bencher.iter(|| numerical::neon::dot_product(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

fn bench_matrix_vector_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix-Vector Multiplication");

    // Matrix sizes (N x N)
    let matrix_sizes: &[usize] = &[64, 128, 256, 512, 1024];

    for &n in matrix_sizes {
        let matrix: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.001).collect();
        let vector: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut result = vec![0.0f32; n];

        group.bench_with_input(BenchmarkId::new("Scalar", n), &n, |bencher, _| {
            bencher.iter(|| numerical::scalar::matrix_vector_mul(black_box(&matrix), black_box(&vector), black_box(&mut result), black_box(n), black_box(n)));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", n), &n, |bencher, _| {
            bencher.iter(|| numerical::portable_simd::matrix_vector_mul(black_box(&matrix), black_box(&vector), black_box(&mut result), black_box(n), black_box(n)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", n), &n, |bencher, _| {
            bencher.iter(|| numerical::neon::matrix_vector_mul(black_box(&matrix), black_box(&vector), black_box(&mut result), black_box(n), black_box(n)));
        });
    }

    group.finish();
}

// =============================================================================
// Data Validation Benchmarks
// =============================================================================

fn bench_range_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("Range Check");

    for &size in SIZES {
        // All values in range [0, 100)
        let data: Vec<i32> = (0..size).map(|i| (i % 100) as i32).collect();

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, _| {
            bencher.iter(|| validation::scalar::all_in_range(black_box(&data), black_box(0), black_box(100)));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", size), &size, |bencher, _| {
            bencher.iter(|| validation::portable_simd::all_in_range(black_box(&data), black_box(0), black_box(100)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", size), &size, |bencher, _| {
            bencher.iter(|| validation::neon::all_in_range(black_box(&data), black_box(0), black_box(100)));
        });
    }

    group.finish();
}

fn bench_is_sorted(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sorted Check");

    for &size in SIZES {
        // Already sorted data
        let data: Vec<i32> = (0..size as i32).collect();

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, _| {
            bencher.iter(|| validation::scalar::is_sorted(black_box(&data)));
        });

        group.bench_with_input(BenchmarkId::new("std::simd", size), &size, |bencher, _| {
            bencher.iter(|| validation::portable_simd::is_sorted(black_box(&data)));
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("NEON", size), &size, |bencher, _| {
            bencher.iter(|| validation::neon::is_sorted(black_box(&data)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rgb_to_grayscale,
    bench_audio_volume,
    bench_audio_mix,
    bench_count_byte,
    bench_find_byte,
    bench_dot_product,
    bench_matrix_vector_mul,
    bench_range_check,
    bench_is_sorted,
);
criterion_main!(benches);
