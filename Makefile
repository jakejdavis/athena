# GENERATION

generate_mnist_conv:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/athena/athena.json

generate_fashionmnist:
	@python main.py generate fashionmnist --specific-output 0 --verbose --additional-config config/athena/athena.json

generate_imdb:
	@python main.py generate imdb --verbose --specific-output 0 --additional-config config/athena/athena.json

generate_mnist_conv_mp:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/athena/athena_multiprocessing.json

generate_mnist_conv_mp_random:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/athena/athena_random_multiprocessing.json

generate_fashionmnist_mp:
	@python main.py generate fashionmnist --specific-output 0 --verbose --additional-config config/athena/athena_multiprocessing.json

generate_imdb_mp:
	@python main.py generate imdb --verbose --specific-output 0 --additional-config config/athena/athena_multiprocessing.json

generate_all_mp:
	@make generate_mnist_conv_mp 
	@make generate_fashionmnist_mp
	@make generate_imdb_mp

# RUN

run_mnist_conv_mp:
	@python main.py run mnist_conv classification --specific-output 0 --verbose --additional-config config/athena/athena_multiprocessing.json
	
run_fashionmnist_mp:
	@python main.py run fashionmnist classification --specific-output 0 --verbose --additional-config config/athena/athena_multiprocessing.json

run_imdb_mp:
	@python main.py run imdb classification --specific-output 0 --verbose --additional-config config/athena/athena_multiprocessing.json

run_all_mp:
	@make run_mnist_conv_mp
	@make run_fashionmnist_mp
	@make run_imdb_mp

# EVALUATION

evaluate_fashionmnist_mp_specific:
	@python main.py evaluate fashionmnist --specific-output $(specific-output) --verbose --additional-config config/athena/athena_multiprocessing.json

evaluate_mnist_conv_mp_specific:
	@python main.py evaluate mnist_conv --specific-output $(specific-output) --verbose --additional-config config/athena/athena_multiprocessing.json

evaluate_imdb_mp_specific:
	@python main.py evaluate imdb --specific-output $(specific-output) --verbose --additional-config config/athena/athena_multiprocessing_no_generic.json

evaluate_all_mp_specific:
	@make evaluate_mnist_conv_mp specific-output=0
	@make evaluate_fashionmnist_mp specific-output=0
	@make evaluate_imdb_mp specific-output=0
	@make evaluate_mnist_conv_mp specific-output=1
	@make evaluate_fashionmnist_mp specific-output=1
	@make evaluate_imdb_mp specific-output=1



evaluate_fashionmnist_mp_fast:
	@python main.py evaluate fashionmnist --verbose --additional-config config/athena/athena_multiprocessing_fast.json

evaluate_mnist_conv_mp_fast:
	@python main.py evaluate mnist_conv --verbose --additional-config config/athena/athena_multiprocessing_fast.json

evaluate_imdb_mp_fast:
	@python main.py evaluate imdb --verbose --additional-config config/athena/athena_multiprocessing_fast.json

evaluate_all_mp_fast:
	@make evaluate_mnist_conv_mp_fast
	@make evaluate_fashionmnist_mp_fast
	@make evaluate_imdb_mp_fast



evaluate_fashionmnist_mp:
	@python main.py evaluate fashionmnist --verbose --additional-config config/athena/athena_multiprocessing.json

evaluate_mnist_conv_mp:
	@python main.py evaluate mnist_conv --verbose --additional-config config/athena/athena_multiprocessing.json

evaluate_imdb_mp:
	@python main.py evaluate imdb --verbose --additional-config config/athena/athena_multiprocessing.json

evaluate_all_mp:
	@make evaluate_mnist_conv_mp
	@make evaluate_fashionmnist_mp
	@make evaluate_imdb_mp


# ENVIROMENTS

env_mac:
	CONDA_SUBDIR=osx-arm64 conda create -n athena python=3.9
	conda activate athena
	pip install -r requirements.macos.txt

env: 	
	conda create -n athena python=3.9
	conda activate athena
	pip install -r requirements.txt


autotype:
	monkeytype run main.py run fashionmnist fashionmnist
	monkeytype list-modules | xargs -n1 -I{} sh -c 'monkeytype apply {}'

