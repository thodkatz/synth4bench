Project build with uv, a modern python package manager (https://docs.astral.sh/uv/).

Installing uv (or follow the installation setup):
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```shell
git clone
uv install
uv run python main.py
```

Check cli args:
```shell
uv run python main.py --help
```

Under this project, we have also clone the `variant-classification` project:
```shell
git clone https://github.com/sfragkoul/variant-classification
```

Datasets are expected to be under `./datasets`. Model states are saved under `./saved_models`, and figures under `./figures`.
