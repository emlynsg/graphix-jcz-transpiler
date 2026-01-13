"""Nox sessions for running CI checks locally."""

import nox


@nox.session(python="3.12")
def ruff_check(session):
    """Run ruff check."""
    session.install("-r", "requirements.txt", "-r", "requirements-dev.txt")
    session.run("ruff", "check")


@nox.session(python="3.12")
def ruff_format(session):
    """Run ruff format check."""
    session.install("-r", "requirements.txt", "-r", "requirements-dev.txt")
    session.run("ruff", "format", "--check")


@nox.session(python="3.12")
def mypy(session):
    """Run mypy type checking."""
    session.install("-r", "requirements.txt", "-r", "requirements-dev.txt")
    session.run("mypy", ".")


@nox.session(python="3.12")
def pytest(session):
    """Run pytest."""
    session.install("-r", "requirements.txt", "-r", "requirements-dev.txt")
    session.run("pytest")


@nox.session(python="3.12")
def ci(session):
    """Run all CI checks."""
    session.install("-r", "requirements.txt", "-r", "requirements-dev.txt")
    session.run("ruff", "check")
    session.run("ruff", "format", "--check")
    session.run("mypy", ".")
    session.run("pytest")
