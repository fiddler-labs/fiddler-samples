SHELL := /bin/bash
SERVICE_NAME=fiddler_samples
FIDDLER_DIR:=$(shell cd . && pwd)

UID:=$(shell id -u)
GID:=$(shell id -g)

default: build stop run logs

build:
	docker build -f dockerfile -t fiddler_samples .

stop:
	-docker stop fiddler_samples

logs:
	docker logs fiddler_samples

run:
	docker run --rm -d --user ${UID}:${GID} --group-add users -p 8888:7100 -v ${FIDDLER_DIR}/content_root:/app/fiddler_samples --name=fiddler_samples fiddler_samples:latest

ssh:
	docker exec -it fiddler_samples /bin/bash

