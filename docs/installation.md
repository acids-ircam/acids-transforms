# Installation

acids_transforms needs Python 3.8 or later, with `torch>=1.10` and `torchaudio>=0.10`.

## From source

```bash
git clone https://github.com/domkirke/acids_transforms.git
cd acids_transforms
pip install .
```

## For development

An editable install with the test dependencies:

```bash
pip install -e ".[test]"
pytest test
```

`test/test_stft_audio.py` writes audio examples of every STFT inversion path into `test/reconstructions/stft/`, so
you can listen to them. `test/test_nn_tilde.py` needs [nn_tilde](https://github.com/acids-ircam/nn_tilde) and
exports test models for Max/MSP and PureData (see {doc}`tutorials/torchscript`).

## Building this documentation

```bash
pip install -r docs/requirements.txt
pip install -e .
make -C docs html          # then open docs/_build/html/index.html
```

Read the Docs builds the same pages for every push to the repository, using `.readthedocs.yaml`.
