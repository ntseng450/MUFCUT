  #!/bin/bash

docker build \
    --network host \
    -t mufcut:1.0.0 \
    ./docker