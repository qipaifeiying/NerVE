docker run --gpus all -it --rm \
    -v $(pwd):/NerVE \
    -w /NerVE \
    nerve_ready:v1 \
    /bin/bash
