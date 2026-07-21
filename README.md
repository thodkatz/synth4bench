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

Datasets are expected to be under `./datasets`. Model states are saved under `./saved_models`, and figures under `./figures`.

# 2vs3 classes

The master branch classifies TP and FP. The branch `three-classes` integrates FN as well, as the class of type C for the xgboost to predict.

The inclusion of FN requires ground truth knowledge of the mutations, and can't be used post-vcf, since there isn't vcf of FN in the first place.
