# Performance Optimization Guide

This guide explains how to use the parallel processing features to maximize performance on your local machine.

## Overview

The parallelized version provides **3-8x speedup** on multi-core machines by:
- Processing topics across multiple CPU cores
- Optimized batch sizes for sentiment analysis and classification
- Memory-efficient chunking
- Real-time progress tracking

## Quick Start

### Default (Automatic Configuration)
The system automatically detects your CPU and memory and configures optimal settings:

```bash
# Uses parallel processing with auto-detected optimal settings
python cli_analysis.py --companies AAPL,MSFT,GOOGL --limit 500
```

### Sequential Mode (Disable Parallelization)
If you encounter issues or want to compare performance:

```bash
# Disable parallel processing
python cli_analysis.py --companies AAPL,MSFT --limit 500 --no-parallel
```

## Advanced Configuration

### Custom Worker Count
Control the number of parallel workers:

```bash
# Use 4 worker processes
python cli_analysis.py --companies AAPL,MSFT --limit 500 --workers 4

# Use maximum available CPUs
python cli_analysis.py --companies AAPL,MSFT --limit 500 --workers 8
```

**Recommendation:** Set workers to `CPU_COUNT - 1` to leave one core for system tasks.

### Custom Batch Sizes
Tune batch sizes for your GPU/CPU:

```bash
# Larger batches for better GPU utilization (if you have a GPU)
python cli_analysis.py --companies AAPL,MSFT --limit 500 \
  --sentiment-batch-size 32 \
  --classification-batch-size 64

# Smaller batches for limited memory
python cli_analysis.py --companies AAPL,MSFT --limit 500 \
  --sentiment-batch-size 4 \
  --classification-batch-size 8
```

### Combined Configuration
```bash
python cli_analysis.py --companies AAPL,MSFT,GOOGL \
  --limit 1000 \
  --workers 6 \
  --sentiment-batch-size 16 \
  --classification-batch-size 32 \
  --write-to-bq
```

## System-Specific Recommendations

### High-End System (8+ CPUs, 16+ GB RAM)
```bash
python cli_analysis.py --companies AAPL,MSFT,GOOGL \
  --mode full \
  --workers 7 \
  --sentiment-batch-size 16 \
  --classification-batch-size 32
```

**Expected Performance:** ~100-200 segments/second

### Mid-Range System (4-8 CPUs, 8-16 GB RAM)
```bash
python cli_analysis.py --companies AAPL,MSFT \
  --limit 500 \
  --workers 3 \
  --sentiment-batch-size 8 \
  --classification-batch-size 16
```

**Expected Performance:** ~50-100 segments/second

### Low-End System (2-4 CPUs, 4-8 GB RAM)
```bash
python cli_analysis.py --companies AAPL \
  --limit 200 \
  --workers 1 \
  --sentiment-batch-size 4 \
  --classification-batch-size 8
```

**Expected Performance:** ~20-50 segments/second

## Benchmarking

Test performance on your system:

```bash
# Quick benchmark (2 runs of 100 records)
python benchmark_parallel.py --companies AAPL,MSFT --limit 100

# Comprehensive benchmark (5 runs of 500 records)
python benchmark_parallel.py --companies AAPL,MSFT,GOOGL --limit 500 --runs 5
```

The benchmark will show:
- Sequential processing time
- Parallel processing time
- Speedup factor
- Time saved percentage

## Performance Metrics

The CLI now shows detailed performance metrics:

```
================================================================================
[SUCCESS] Analysis Complete!
================================================================================

Output:
  File: outputs/cli_analysis_results_20250217_120000.csv
  Size: 15.32 MB

Statistics:
  Total records: 12,450
  Transcripts: 25
  Companies: 3
  Topics detected: 8,234

Performance:
  Time taken: 2m 15s
  Processing rate: 92.3 segments/sec
================================================================================
```

## Progress Tracking

The parallel version shows real-time progress:

```
[PARALLEL PROCESSING MODE]
================================================================================

1. Classification Stage (1000 segments)
--------------------------------------------------------------------------------
   Running batch interaction classification...
   Running batch role classification...

2. Topic & Sentiment Analysis Stage
--------------------------------------------------------------------------------
   Processing 1000 texts in 40 chunks across 4 workers...
Topic Detection: 100%|███████████████████████| 40/40 [00:15<00:00]
   Running batch sentiment analysis for 2340 topic pairs...
Sentiment Analysis: 100%|███████████████████| 293/293 [00:42<00:00]
   Completed analysis in 58.3s (17.2 texts/sec)

3. Results Assembly Stage
--------------------------------------------------------------------------------
   Assembling 1000 segments into enriched results...
Assembling Results: 100%|██████████████████| 1000/1000 [00:02<00:00]
```

## Troubleshooting

### Memory Errors
If you see "Out of Memory" errors:

1. Reduce batch sizes:
   ```bash
   --sentiment-batch-size 4 --classification-batch-size 8
   ```

2. Reduce workers:
   ```bash
   --workers 1
   ```

3. Process smaller chunks:
   ```bash
   --limit 200
   ```

### Slow Performance
If parallel is slower than sequential:

1. Check if you're running on battery power (laptops throttle CPUs)
2. Close other applications
3. Ensure you have enough RAM available
4. Try reducing workers to avoid context switching overhead

### GPU Not Being Used
The models run on CPU by default. To use GPU:

1. Ensure you have `torch` with CUDA support installed
2. The models will automatically use GPU if available
3. Increase batch sizes to better utilize GPU:
   ```bash
   --sentiment-batch-size 32 --classification-batch-size 64
   ```

## Cloud Run Considerations

When using `--cloud`, the parallelization happens on the Cloud Run instance. The cloud service uses:
- Auto-configured batch sizes based on Cloud Run instance specs
- Sequential processing to avoid GIL and multiprocessing overhead in containerized environment
- Optimized for throughput over individual request latency

For local development and testing, parallel processing is recommended.
For production large-scale processing, Cloud Run is recommended.

## Best Practices

1. **Test with small limits first**: Always test with `--limit 100` before running full analysis
2. **Monitor resource usage**: Use Task Manager (Windows) or `htop` (Linux/Mac) to monitor CPU and memory
3. **Optimize for your data**: If analyzing few companies, use fewer workers
4. **Use --no-content for speed**: Exclude transcript content from output if not needed
5. **Benchmark your system**: Run `benchmark_parallel.py` to find optimal settings

## Example Workflows

### Development/Testing
```bash
# Fast iteration with parallel processing
python cli_analysis.py --companies AAPL --limit 50 --no-content
```

### Production Run (Local)
```bash
# Full analysis with optimal settings
python cli_analysis.py \
  --mode full \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --workers 6 \
  --sentiment-batch-size 16 \
  --classification-batch-size 32 \
  --write-to-bq
```

### Production Run (Cloud)
```bash
# Leverage Cloud Run for large datasets
python cli_analysis.py \
  --mode full \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --cloud
```

## Getting Help

- Run `python cli_analysis.py --help` to see all options
- Check system configuration: Review output of `get_optimal_config()` on startup
- Report issues: If parallel processing doesn't work as expected, use `--no-parallel` and report the issue
