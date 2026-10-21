IMAGE_NAME := caddy-webserver
CONTAINER_NAME := caddy-webserver-container
PORTS := -p 8080:8080 -p 443:443

build:
	docker build -t $(IMAGE_NAME) -f Dockerfile .

run:
	docker run -d --name $(CONTAINER_NAME) $(PORTS) $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm -f $(CONTAINER_NAME)

clean: stop
	docker rmi $(IMAGE_NAME) || true

rebuild: clean build run

ps:
	docker ps

ps-all:
	docker ps -a

logs:
	docker logs $(CONTAINER_NAME)

shell:
	docker exec -it $(CONTAINER_NAME) sh
