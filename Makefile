generate_mnist_conv:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/arachne/arachne.json

generate_fashionmnist:
	@python main.py generate fashionmnist --specific-output 0 --verbose --additional-config config/arachne/arachne.json

generate_boston:
	@python main.py generate boston --verbose --additional-config config/arachne/arachne.json

generate_mnist_conv_mp:
	@python main.py generate mnist_conv --specific-output 0 --verbose --additional-config config/arachne/arachne_multiprocessing.json > /dev/null

generate_fashionmnist_mp:
	@python main.py generate fashionmnist --specific-output 0 --verbose --additional-config config/arachne/arachne_multiprocessing.json > /dev/null

generate_boston_mp:
	@python main.py generate boston --verbose --additional-config config/arachne/arachne_multiprocessing.json > /dev/null

generate_all_mp:
	@make generate_mnist_conv_mp 
	@make generate_fashionmnist_mp
	@make generate_boston_mp

autotype:
	monkeytype run main.py run fashionmnist fashionmnist
	monkeytype list-modules | xargs -n1 -I{} sh -c 'monkeytype apply {}'