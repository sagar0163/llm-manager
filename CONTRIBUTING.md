# Contributing to LLM Manager

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/sagar0163/llm-manager.git
cd llm-manager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Making Changes

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m "feat: add new feature"`
6. Push to main: `git push origin main`

## Commit Messages

Follow conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `refactor:` code refactoring
- `test:` adding tests

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Keep functions small and focused

---

⭐ Star us if you like this project!
