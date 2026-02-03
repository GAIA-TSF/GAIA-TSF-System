# Contributing

## Recommended Software Development Workflow

Coding style [PEP 8 (Style Guide for Python Code)]:

- Class names: `CapWords`
- Class properties/variables: `snake_case`
- Class methods/function names: `snake_case`
- Protected properties/methods: `_snake_case`
- Private properties/methods: `__snake_case`
- Package names should be short, all-lowercase, and preferably without underscores.
- 2 blank lines before a class definition
- 1 blank line before a function definition
- trailing comma at the last argument of one-arg-per-line-formatted func calls
- single quotes for string definitions unless the string already contains them

Branching Strategy

- `main`: Production-ready code (matches CDR baseline).
- `develop`: Integration branch for the current sprint.
- `feature/*`: Individual tasks.

Collaboration and Code Review

- create PR for completed features or bugfix (base branch: `develop`)
- conduct peer code reviews
- use comments and suggestions to improve code quality
- resolve conflicts and ensure code consistency
- merge PR only when approved
- optionally use GitHub Projects to break into tasks, features, and bug fixes

Issue Tracking

- use GitHub Issues to manage tasks and bugs
- assign issues, set labels, milestones, and priorities
- link commits and PRs to issues

Continuous Integration (CI)

- configure GitHub Actions for automatic testing
- run unit tests and static analysis on every push and PR
- prevent merging if tests fail
- code quality check tests open automated PRs fixing the violations - merge them

Documentation

- update README with usage instructions
- document code with docstrings and comments

Release Management

- tag stable versions using semantic versioning
- create GitHub Releases with release notes

## Programming languange and style

- Python
- Object oriented paradigm
- tests driven by `pytest` package

## Deployment strategy

- use a unified deployment system (Docker?)
- use a unified base image (python-3.14?)
- use pre-built images for CI

## Recommended code subsystem layout

Example for Earth Observation Data Uploader (`earth_observation_data_uploader`):

```
earth_observation_data_uploader/
├── README.md
├── __init__.py
├── data_acquisition_gateway
│   ├── __init__.py
├── data_extraction
│   ├── __init__.py
├── manual_file_loader
│   ├── __init__.py
├── docker
│   ├── Dockerfile
│   └── requirements.txt
└── tests
    ├── test_interfaces.py
    ├── test_modules.py
    └── test_subsystem.py
```
