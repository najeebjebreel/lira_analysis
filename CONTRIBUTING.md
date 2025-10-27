# Contributing to MIA Research

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

### Setting Up Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/najeebjebreel/lira_analysis.git
   cd lira_analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### 1. Create a Branch

Create a feature branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clear, documented code
- Follow the existing code style
- Add tests for new features
- Update documentation as needed

### 3. Run Tests

Before submitting, ensure all tests pass:
```bash
pytest tests/ -v
```

### 4. Format Code

Format your code with black and isort:
```bash
black src/ tests/
isort src/ tests/
```

### 5. Lint Code

Check for issues with flake8:
```bash
flake8 src/ tests/ --max-line-length=100
```

### 6. Commit Changes

Write clear, descriptive commit messages:
```bash
git add .
git commit -m "Add feature: brief description"
```

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all functions, classes, and modules
- Format with black and isort

### Docstring Format
Use Google-style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of function.

    Longer description if needed.

    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2

    Returns:
        type: Description of return value

    Raises:
        ExceptionType: When this exception is raised
    """
    pass
```

## Testing Guidelines

### Writing Tests
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures defined in `tests/conftest.py`
- Aim for high code coverage

### Test Structure
```python
def test_feature_name():
    """Test description."""
    # Arrange
    setup_data = create_test_data()

    # Act
    result = function_to_test(setup_data)

    # Assert
    assert result == expected_value
```

## Adding New Features

### New Datasets
1. Add loader function in `src/mia_research/data/data_utils.py`
2. Add dataset statistics to `DATASET_STATS` dictionary
3. Update documentation in README.md
4. Add tests in `tests/test_data_utils.py`

### New Attack Methods
1. Create new file in `src/mia_research/attacks/`
2. Implement attack class following LiRA pattern
3. Add to `__init__.py` exports
4. Update configuration files
5. Add tests in `tests/`
6. Document in README.md

### New Models
1. Add model in `src/mia_research/models/model_utils.py`
2. Update `get_model()` function
3. Add tests in `tests/test_models.py`
4. Document supported architectures in README.md

## Documentation

### Update Documentation When:
- Adding new features
- Changing existing behavior
- Adding new configuration options
- Fixing bugs that affect user-facing behavior

### Documentation Locations:
- **README.md**: Installation, usage, features
- **Docstrings**: All functions, classes, modules
- **CHANGELOG.md**: All changes for each version
- **docs/**: Detailed guides and tutorials

## Pull Request Guidelines

### Before Submitting
- [ ] All tests pass
- [ ] Code is formatted (black, isort)
- [ ] Code passes linting (flake8)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Commit messages are clear

### PR Description Should Include:
- What changes were made
- Why the changes were necessary
- Any breaking changes
- Related issue numbers (if applicable)

### Review Process
1. Automated tests will run on your PR
2. Maintainers will review your code
3. Address any requested changes
4. Once approved, your PR will be merged

## Reporting Issues

### Bug Reports
Include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, package versions)
- Error messages and stack traces

### Feature Requests
Include:
- Clear description of the feature
- Use cases and motivation
- Proposed implementation (if you have ideas)

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome diverse perspectives
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Unacceptable Behavior
- Harassment or discriminatory language
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

## Questions?

If you have questions about contributing, please open an issue with the "question" label.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
