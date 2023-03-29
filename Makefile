generate_mnist_conv:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/athena/athena.json

generate_fashionmnist:
	@python main.py generate fashionmnist --specific-output 0 --verbose --additional-config config/athena/athena.json

generate_boston:
	@python main.py generate boston --verbose --additional-config config/athena/athena.json

generate_mnist_conv_mp:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/athena/athena_multiprocessing.json > /dev/null

generate_mnist_conv_mp_random:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/athena/athena_random_multiprocessing.json > /dev/null

generate_fashionmnist_mp:
	@python main.py generate fashionmnist --specific-output 0 --verbose --additional-config config/athena/athena_multiprocessing.json > /dev/null

generate_boston_mp:
	@python main.py generate boston --verbose --additional-config config/athena/athena_multiprocessing.json > /dev/null

generate_all_mp:
	@make generate_mnist_conv_mp 
	@make generate_fashionmnist_mp
	@make generate_boston_mp
	
run_fashionmnist_mp:
	@python main.py run fashionmnist classification --specific-output 0 --verbose --additional-config config/athena/athena_multiprocessing.json > /dev/null


make_figures:
	pyreverse models -o puml --output-directory ./figures
	cat figures/classes.puml | java -jar plant/plantuml.jar -tpdf -pipe > figures/models.pdf
	pyreverse operators -o puml --output-directory ./figures
	cat figures/classes.puml | java -jar plant/plantuml.jar -tpdf -pipe > figures/operators.pdf

autotype:
	monkeytype run main.py run fashionmnist fashionmnist
	monkeytype list-modules | xargs -n1 -I{} sh -c 'monkeytype apply {}'
