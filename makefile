SHELL := /bin/bash
SERVICE_NAME=fiddler_samples
FIDDLER_DIR:=$(shell cd . && pwd)

default: build stop run logs

build:
	docker build -f dockerfile -t fiddler_samples .

stop:
	-docker stop fiddler_samples

logs:
	docker logs fiddler_samples

run:
	docker run --rm -d -p 7100:7100 -v ${FIDDLER_DIR}/content_root:/app/fiddler_samples --name=fiddler_samples fiddler_samples:latest

ssh:
	docker exec -it fiddler_samples /bin/bash

