# Docker Commands

## Docker Basic
- `docker images` : Shows locally available images
- `docker rmi IMAGE_NAME` : Removes the docker image
- `docker pull alphine` : Pull alphine image from hub.docker.com
- `docker image inspect IMAGE_ID` : Inspect the settings of an image
- `docker pull registry.com:5000/testing/test-image` : pull image my specific registry
- `docker pull alphine:1.0.0.1` : Pulling specific image
- `docker run alphine sh` : runs sh command on alphine and then quits.
- `docker stop CONTAINER_ID` : stops the specific running container.
- `docker rm CONTAINER_ID` : Removes a container
- `docker container ps` : Shows running docker containers
- `docker commit CONTAINER_ID NEW_IMAGE_NAME` : commites the changes made in the container and saves it as new image
- `docker build -t IMAGE_NAME:TAG /path/to/Dockerfile` : build the docker image with the given image
- `docker tag IMAGE:TAG IMAGE:TAG` : Adds a new tagged image
- `docker login -u USERNAME -p KEY` : Login to private docker registry
- `docker push IMAGE_NAME:TAG` : Pushes the image to the logged in registry 

## Docker running
- `docker container ps` : get the status of running containers
- `docker ps -a` : shows all containers
- `docker run -it alphine` : Runs interactive terminal on alphine image
- `docker run -d nginx` : runs ngnix image in detached mode
- `docker run -d -v /source/in/host:/dest/in/contianer alphine -p80:80 webserver` : Runs the webserve at port 80 with the volumes specified
- `docker exec CONTAINER_ID ls` : runs ls command on the container specified

## DockerFile
- `FROM IMAGE_ID` : takes the base image
- `MAINTAINER austin@nordstrom.com` : the composer
- `RUN apt-get update` : Updating package manager indexes
- `RUN apt install SERVICE_NAME` : install some images
- `CMD ["systemctl","start nginx"]` :  Starts the nginx server is no other commands as passed with docker run. Like a default case.
- `ENTRYPOINT ["systemctl","start","nginx"]` : starts nginx server no matter how you run. Like a finally case but at start
- `docker build -t IMAGE_NAME:TAG /path/to/Dockerfile` : build the docker image with the given image
- `docker tag IMAGE IMAGE:TAG` : Tags the image
