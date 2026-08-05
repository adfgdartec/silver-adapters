# Contributing to silver-adapters

Thank you for your interest in contributing to silver-adapters! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setting Up Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/adfgdartec/silver-adapters.git
   cd silver-adapters
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install optional dependencies for testing**
   ```bash
   pip install -e ".[all]"  # Or specific: [pytorch], [tensorflow], etc.
   ```

5. **Run tests**
   ```bash
   pytest
   ```

6. **Run tests with coverage**
   ```bash
   pytest --cov=silver_adapters --cov-report=html
   ```

## Code Style

We use the following tools to maintain code quality:

- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

Run all quality checks:
```bash
flake8 src/ tests/
mypy src/
pytest
```

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, descriptive commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Handle optional dependencies gracefully

3. **Run tests**
   ```bash
   pytest
   ```

4. **Submit a pull request**
   - Describe your changes clearly
   - Reference any related issues
   - Ensure all tests pass
   - Note any optional dependencies required

## Testing

We aim for high test coverage. When adding new features:

- Write unit tests for new adapter functions
- Mock optional dependencies (torch, tensorflow, etc.)
- Test error handling for missing dependencies
- Ensure existing tests still pass

### Test Structure
```
tests/
├── __init__.py
└── test_adapters.py
```

## Optional Dependencies

This package has optional dependencies for different ML frameworks:

- `pytorch` - PyTorch support
- `tensorflow` - TensorFlow support
- `keras` - Keras support
- `kaggle` - Kaggle API support
- `jupyter` - Jupyter notebook support

When adding new framework-specific features:

1. Add the dependency to `pyproject.toml` under `project.optional-dependencies`
2. Import the dependency inside the function (not at module level)
3. Provide clear error messages when the dependency is missing
4. Add tests that mock the dependency

## Documentation

- Update docstrings for any modified functions
- Add examples for new adapters
- Update README.md if user-facing changes are made
- Document any new optional dependencies

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a GitHub release
4. Package will be automatically published to PyPI

## Questions?

Feel free to open an issue for questions or discussions about contributions.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
