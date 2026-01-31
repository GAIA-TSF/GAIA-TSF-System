# Contributing

## Recommended Software Development Workflow

Coding style [PEP 8 (Style Guide for Python Code)]:

- Class names: `CapWords`
- Class properties/variables: `snake_case`
- Class methods/function names: `snake_case`
- Protected properties/methods: `_snake_case`
- Private properties/methods: `__snake_case`
- Package names should be short, all-lowercase, and preferably without underscores.

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

- use a unified deployment system (Docker)
- use a unified base image (python-3.14?)
- use pre-built images for CI

## Recommended code subsystem layout

Subsystems code is placed in `subsystems` directory. Subsystems are
defined in separate sub-directories, which are named according to the
subsystem abbreviation.

Recommended layout for a single subsystem:

```
subsystem_name_abbreviated
├── __init__.py
├── module_name_1
    ├── __init__.py
├── module_name_2
    ├── __init__.py
├── README.md
└── tests
    ├── test_interfaces.py
    ├── test_modules.py
    └── test_subsystem.py
```

The `__init__.py` file defines the subsystem class carrying its full
(unabbreviated) name. Individual subsystem modules are defined in
sub-directories. Each of these sub-directories contains an
`__init__.py` file with the definition of the corresponding module
class. The names of the module sub-directories and the module classes
are derived from the full (unabbreviated) module names.

The `tests` directory contains three separate files. The
`tests_subsystem.py` contains tests focused on the subsystem as a
whole. The `test_modules.py` file contains unit tests targeting
individual components of the subsystem. The last file,
`test_interfaces.py`, contains integration tests that verify the
integration of individual modules within the subsystem and within the
system as a whole.

There is also a `README.md` file containing information about the
subsystem, including a figure describing the design of the subsystem
and its individual modules.

Here is a minimalistic example for *Earth Observation Data Uploader*
subsystem:

```
eou/
├── __init__.py     # defines EarthObservationDataUploader class
├── data_acquisition_gateway
│   ├── __init__.py # defines DataAcquisitionGateway class
├── data_extraction
│   ├── __init__.py # defines DataExtraction class
├── manual_file_loader
│   ├── __init__.py # defines ManualFileLoader
├── README.md
└── tests
    ├── test_interfaces.py
    ├── test_modules.py
    └── test_subsystem.py
```
