let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };
  python = pkgs.python311;
  pythonPackages = python.pkgs;
  lib-path = with pkgs; lib.makeLibraryPath [
    graphviz
    libffi
    openssl
    stdenv.cc.cc
  ];
in with pkgs; mkShell {
  packages = [
    # Necessary
    pythonPackages.einops
    pythonPackages.graphviz
    pythonPackages.h5py
    pythonPackages.librosa
    pythonPackages.matplotlib
    pythonPackages.psutil
    pythonPackages.pympler
    pythonPackages.rich
    pythonPackages.scipy
    pythonPackages.seaborn
    pythonPackages.scikit-learn
    pythonPackages.torch
    pythonPackages.torchaudio
    pythonPackages.torchvision

    # Jupyter Lab
    pythonPackages.ipython
    pythonPackages.jupyter
  ];

  buildInputs = [
    graphviz
    readline
    libffi
    openssl
    git
    openssh
    rsync
  ];

  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
    VENV=.venv

    if test ! -d $VENV; then
      python3.11 -m venv $VENV
    fi
    source ./$VENV/bin/activate
    export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
    # pip install -r requirements.txt

    # Temporary for script
    export CONFIG_FILE=configs/einspace_quick/evolution_test/addnist/evolution.yaml
    export GPU_ID=0
    export LOG_FILE="logs/$(sed -r 's_configs/__;s_\.yaml$__' <<< $CONFIG_FILE)"
  '';

  postShellHook = ''
    ln -sf ${python.sitePackages}/* ./.venv/lib/python3.11/site-packages
  '';
}
