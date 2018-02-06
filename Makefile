NAME=mlcollege/image-processing

all: download build run

run:
	docker run -ti -p 9997:8000 -p 22239:6006 -p 22240:6007 -v $(shell pwd)/notebooks:/notebooks -v $(shell pwd)/tensorboard_summaries_mnt:/tensorboard_summaries_mnt $(NAME)

stop:
	docker stop $(NAME)

build:
	docker build -t $(NAME) .

push:
	docker push $(NAME)

pull:
	docker pull $(NAME)
