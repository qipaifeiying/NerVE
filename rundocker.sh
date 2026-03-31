docker run --gpus all -it --rm \
    -v $(pwd):/NerVE \
    -w /NerVE \
    nerve:v2 \
    /bin/bash
