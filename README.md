# BLINK_api

### Start entity_linking docker container

docker build -t entity_linking:latest .

docker run --gpus all -p 8080:5000 -v ${PWD}/entity_linking_container/models:/blink/models --name entity_linking entity_linking
