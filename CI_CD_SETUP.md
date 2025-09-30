# CI/CD Pipeline Documentation

## Overview

This project uses a comprehensive CI/CD pipeline with GitHub Actions to ensure code quality, testing, and reliable deployments. The pipeline consists of multiple workflows that run on different triggers.

## Workflows

### 1. Continuous Integration (`ci.yml`)
**Triggers:** Push to `main`/`develop` branches, Pull Requests

**Jobs:**
- **Test Matrix:** Runs on Python 3.10, 3.11, and 3.12
- **Code Quality:** Linting (ruff), formatting (black, ruff), type checking (mypy)
- **Testing:** Full test suite with coverage reporting
- **Docker:** Build and test Docker image

**Quality Gates:**
- ✅ All tests must pass
- ✅ Code coverage reporting to Codecov
- ✅ Linting and formatting compliance
- ✅ Type checking validation
- ✅ Docker build success

### 2. Pull Request Quality Checks (`pr-quality-checks.yml`)
**Triggers:** Pull Requests only

**Enhanced Features:**
- 📊 Detailed step summaries in PR
- 🔒 Security scanning with bandit
- 📈 Coverage badge generation
- 🎯 Focused feedback for developers

## Code Quality Tools

### Ruff (Linting & Formatting)
```bash
# Check code
ruff check .

# Fix auto-fixable issues  
ruff check . --fix

# Format code
ruff format .
```

**Configuration:** `pyproject.toml`
- Line length: 88 characters
- Target: Python 3.8+
- Rules: pycodestyle, Pyflakes, isort, flake8-bugbear, comprehensions, pyupgrade

### Black (Code Formatting)
```bash
# Check formatting
black --check .

# Apply formatting
black .
```

### MyPy (Type Checking)
```bash
# Type check the codebase
mypy tier1_suite/
```

**Configuration:** Strict type checking enabled with comprehensive warnings

### Pre-commit Hooks
Install and setup local pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

**Hooks:**
- Trailing whitespace removal
- End-of-file fixing
- YAML/TOML validation
- Black formatting
- Ruff linting
- MyPy type checking
- Pytest execution

## Testing Framework

### Test Categories
1. **Unit Tests** (`test_transforms.py`): Individual component testing
2. **Integration Tests**: Cross-component validation  
3. **Leakage Tests** (`test_leakage.py`): Data leakage prevention
4. **Determinism Tests** (`test_determinism.py`): Reproducibility validation
5. **Schema Tests** (`test_schemas.py`): I/O validation

### Running Tests Locally
```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=tier1_suite --cov-report=html

# Specific test categories
pytest tests/ -m unit
pytest tests/ -m leakage  
pytest tests/ -m determinism
pytest tests/ -m schemas

# Performance tests
pytest tests/ -m "not slow"
```

### Coverage Requirements
- **Minimum:** 80% coverage
- **Target:** 90%+ coverage
- **Reports:** HTML and XML formats generated

## Development Workflow

### 1. Local Development
```bash
# Setup development environment
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run quality checks locally
ruff check .
black --check .
mypy tier1_suite/
pytest tests/ -v
```

### 2. Pull Request Process
1. **Create Feature Branch:** `git checkout -b feature/your-feature`
2. **Develop & Test:** Local development with pre-commit hooks
3. **Push & Create PR:** GitHub automatically runs quality checks
4. **Review Results:** Check PR quality report and address issues
5. **Merge:** Only after all checks pass

### 3. Release Process
- **Main Branch:** Production-ready code
- **Develop Branch:** Integration branch for features
- **Automatic:** Docker images built and tested on merge

## Troubleshooting

### Common Issues

**Ruff Formatting Errors:**
```bash
ruff format .  # Auto-fix formatting
```

**Type Checking Failures:**
```bash
mypy tier1_suite/ --ignore-missing-imports  # Check with relaxed rules
```

**Test Failures:**
```bash
pytest tests/ -v --tb=long  # Verbose failure information
```

**Coverage Too Low:**
```bash
pytest tests/ --cov=tier1_suite --cov-report=html
# Open htmlcov/index.html to see uncovered lines
```

### Performance Optimization
- **Parallel Testing:** `pytest -n auto` (with pytest-xdist)
- **Test Selection:** Use markers to run subset of tests
- **Cache:** GitHub Actions caches dependencies

## Security

### Automated Security Checks
- **Bandit:** Static security analysis for Python
- **Dependency Scanning:** GitHub Dependabot for vulnerabilities
- **Code Scanning:** GitHub CodeQL analysis

### Manual Security Reviews
- Regular dependency updates
- Security-focused code reviews
- OWASP guidelines compliance

## Monitoring & Metrics

### Coverage Tracking
- **Codecov Integration:** Automated coverage reporting
- **Trend Analysis:** Coverage changes over time
- **PR Impact:** Coverage impact of changes

### Performance Monitoring
- **Test Duration:** Track slow tests with `--durations=10`
- **CI Performance:** Monitor workflow execution times
- **Resource Usage:** Memory and CPU monitoring

## Configuration Files

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Main CI pipeline |
| `.github/workflows/pr-quality-checks.yml` | PR-specific checks |
| `.pre-commit-config.yaml` | Local pre-commit hooks |
| `pyproject.toml` | Tool configuration |
| `pytest.ini` | Test configuration |

## Best Practices

1. **Write Tests First:** TDD approach with comprehensive coverage
2. **Small Commits:** Atomic changes for better review
3. **Descriptive Messages:** Clear commit and PR descriptions  
4. **Code Reviews:** All changes reviewed before merge
5. **Documentation:** Keep docs updated with changes
6. **Performance:** Monitor and optimize slow tests
7. **Security:** Regular dependency and security updates