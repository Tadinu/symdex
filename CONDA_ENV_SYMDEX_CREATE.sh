SYMDEX_ENV=${1:-"./.conda_envs/symdex"}
if [ ! -d "$SYMDEX_ENV" ]
then
	conda env create -f ./conda/env.yaml --prefix $SYMDEX_ENV
	conda config --set env_prompt '({name})'
	conda info --envs
    # conda init # -> This will add auto conda init script to ~/.bashrc
	conda activate $SYMDEX_ENV
else
	echo "$SYMDEX_ENV already exists"
fi
