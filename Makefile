generate_mnist_conv:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/arachne/arachne.yaml

generate_fashionmnist:
	@python main.py generate fashionmnist --specific-output 0 --verbose --additional-config config/arachne/arachne.yaml

generate_boston:
	@python main.py generate boston --verbose --additional-config config/arachne/arachne.yaml

generate_mnist_conv_mp:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/arachne/arachne_multiprocessing.yaml

generate_fashionmnist_mp:
	@python main.py generate fashionmnist --specific-output 0 --verbose --additional-config config/arachne/arachne_multiprocessing.yaml

generate_boston_mp:
	@python main.py generate boston --verbose --additional-config config/arachne/arachne_multiprocessing.yaml

autotype:
	monkeytype run main.py run fashionmnist fashionmnist
	monkeytype list-modules | xargs -n1 -I{} sh -c 'monkeytype apply {}'