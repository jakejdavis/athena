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

evaluate_fashionmnist_mp:
	@python main.py evaluate fashionmnist --specific-output $(specific-output) --verbose --additional-config config/athena/athena_multiprocessing.json

evaluate_mnist_conv_mp:
	@python main.py evaluate mnist_conv --specific-output $(specific-output) --verbose --additional-config config/athena/athena_multiprocessing.json

evaluate_imdb_mp:
	@python main.py evaluate imdb --specific-output $(specific-output) --verbose --additional-config config/athena/athena_multiprocessing_no_trivial.json


evaluate_all_mp:
	@make evaluate_mnist_conv_mp specific-output=0
	@make evaluate_fashionmnist_mp specific-output=0
	@make evaluate_imdb_mp specific-output=0
	@make evaluate_mnist_conv_mp specific-output=1
	@make evaluate_fashionmnist_mp specific-output=1
	@make evaluate_imdb_mp specific-output=1

# PLOT EVALUATION

plot_evaluation:
	@python evaluation/generate_figures.py

# FIGURES

make_figures:
	pyreverse models -o puml --output-directory ./figures
	cat figures/classes.puml | java -jar plant/plantuml.jar -tpdf -pipe > figures/models.pdf
	pyreverse operators -o puml --output-directory ./figures
	cat figures/classes.puml | java -jar plant/plantuml.jar -tpdf -pipe > figures/operators.pdf
	pyreverse mutants.operators.athena.localisers -o puml --output-directory ./figures
	cat figures/classes.puml | java -jar plant/plantuml.jar -tpdf -pipe > figures/localisers.pdf
	pyreverse mutants.operators.athena.searchers -o puml --output-directory ./figures
	cat figures/classes.puml | java -jar plant/plantuml.jar -tpdf -pipe > figures/searchers.pdf

autotype:
	monkeytype run main.py run fashionmnist fashionmnist
	monkeytype list-modules | xargs -n1 -I{} sh -c 'monkeytype apply {}'
