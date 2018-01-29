NAME=mlcollege/image-processing

all: download build run

run:
	docker run -ti -p 9997:8000 -p 22239:6006 -v $(shell pwd)/notebooks:/notebooks -v $(shell pwd)/tensorboard_summaries:/tensorboard_summaries $(NAME)

stop:
	docker stop $(NAME)

build:
	docker build -t $(NAME) .

push:
	docker push $(NAME)

pull:
	docker pull $(NAME)
