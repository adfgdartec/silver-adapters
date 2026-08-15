# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-15

### Changed
- Coordinated the package release with model adapter and checkpoint examples.

## [0.2.0] - 2026-08-15

### Changed
- Added configurable positive HTTP timeouts to `RemoteTrainingClient`.
- Reused owned HTTP sessions across requests and added asynchronous cleanup.
- Added generic PyTorch and optional Hugging Face model adapters for causal-LM and classification tasks.

## [0.1.0] - 2024-08-04

### Added
- Initial release of silver-adapters
- PyTorch bridge utilities for checkpoint management
- TensorFlow/Keras bridge utilities for model saving/loading
- JSONL protocol encoding and decoding for event streaming
- Remote training client with async HTTP support
- Jupyter notebook template generation
- Data import plans for pandas and Kaggle
- Framework command bridge creation
- Comprehensive test suite with mocking for optional dependencies
- Support for Python 3.8-3.12

### Features
- `PyTorchBridge` - PyTorch checkpoint utilities
- `TensorFlowBridge` - TensorFlow/Keras model utilities
- `RemoteTrainingClient` - HTTP client for remote training
- `encode_jsonl()` / `decode_jsonl()` - JSONL protocol handling
- `notebook_document()` / `silver_notebook_cells()` - Jupyter utilities
- `python_framework()` - Framework command bridge creation
- `pandas_import()` / `kaggle_import()` - Data import planning
- Optional dependencies for specific framework support
- Async HTTP operations for remote training

## [Unreleased]
