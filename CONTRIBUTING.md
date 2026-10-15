# Contributing

Issues and pull requests are welcome. Before opening a pull request:

1. Install the development dependencies with `python -m pip install -e '.[dev]'`.
2. Run `python -m pytest`.
3. Keep input spectral libraries, extracted tables, and private analysis outputs
   out of commits. The repository `.gitignore` excludes the common data formats.
4. When changing format decoding, add a small synthetic fixture test. Do not
   redistribute DIA-NN libraries or data from other studies.
