# Python

1. Prerequisites: macos, ubuntu
2. Install Pyenv, Autoenv
3. Install Python 3.11: virtualenv, jupyter, numpy ...

---

## Prerequisites

- pyenv wiki: [Suggested build environment](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)

### macOS

```bash
xcode-select --install
brew install openssl readline sqlite3 xz zlib tcl-tk
```

### Linux

#### Ubuntu

```bash
build_tools="
build-essential
libssl-dev
zlib1g-dev
libbz2-dev
libreadline-dev
libsqlite3-dev
curl
libncursesw5-dev
xz-utils
tk-dev
libxml2-dev
libxmlsec1-dev
libffi-dev
liblzma-dev
"

sudo apt update
sudo apt install $build_tools
```

---

## Install Pyenv, Autoenv

### Set profile.rc

Add to `.bashrc` or `.zshrc`:

```bash
# pyenv
if [ -d "$HOME/.pyenv" ]; then
    export PYENV_ROOT="$HOME/.pyenv"
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi
```

(Option) Set Autoenv:

```bash
# autoenv
if [ -d "$HOME/.autoenv" ]; then
    export AUTOENV_ENV_FILENAME='.autoenv'
    export AUTOENV_ENV_LEAVE_FILENAME='.autoenv.leave'
    export AUTOENV_ENABLE_LEAVE='enabled'
    source "$HOME/.autoenv/activate.sh"
fi
```

### Pyenv

- [pyenv](https://github.com/pyenv/pyenv)

```bash
curl https://pyenv.run | bash
```

### (Option) Autoenv

- [autoenv](https://github.com/hyperupcall/autoenv)

```bash
git clone https://github.com/hyperupcall/autoenv $HOME/.autoenv
```

---

## Install Python 3.11

```bash
version=3.11
latest_version=$(pyenv latest -k $version)
pyenv install -v --skip-existing $latest_version
```

### Set Project Python Virtual Env

```bash
pyenv virtualenv $latest_version deep
```

### (Option) Set .autoenv

```bash
tee .autoenv <<EOF
pyenv activate deep
EOF

tee .autoenv.leave <<EOF
pyenv deactivate
EOF
```

### Activate Python Virtual Env

```bash
pyenv activate deep
pip install --upgrade pip build
```

### Install Packages

```bash
pip install -r requirements.txt
```

or

```bash
pip install jupyterlab
pip install numpy
pip install matplotlib
```
