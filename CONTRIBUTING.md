# Contributing to RayRoom

First off, thank you for considering contributing to RayRoom! It's people like you that make RayRoom such a great tool.

## Where do I go from here?

If you've noticed a bug or have a question, [search the issue tracker](https://github.com/rayroom/rayroom/issues) to see if someone else in the community has already created a ticket. If not, feel free to create a new one!

## Fork & create a branch

If you're looking to contribute code, the first step is to fork this repository. Then, create a new branch from `main` to work in. This is where you'll make your changes.

## Getting Started

To get your development environment set up, you should create a Python virtual environment and install the project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Making Changes

Here are some things to keep in mind as you make your changes:

- **Code Style**: We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. We use `flake8` to check for style issues. Before submitting a pull request, please run `flake8` on your changes.

  ```bash
  flake8 .
  ```

- **Documentation**: If you're adding a new feature or changing an existing one, please update the documentation in the `docs` directory to reflect your changes.

- **Testing**: We encourage you to write tests for your changes. While we don't have a formal testing framework set up yet, you can add tests to the `examples` directory.

## Submitting a Pull Request

When you're ready to submit your changes, create a pull request from your branch to the `main` branch of the main RayRoom repository.

In your pull request description, please provide a clear and concise summary of the changes you've made. If your pull request addresses an open issue, please link to it in the description.

## Code of Conduct

While this project does not have a formal code of conduct yet, we expect all contributors to be respectful and professional. Please be kind and considerate to others.

---

Thank you for contributing to RayRoom!
