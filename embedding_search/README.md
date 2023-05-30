a

### Downloading and Precomputing Embeddings from Laion Chunk

To download a laion-chunk, we first need to download parquet files that contain image URLs. 
For example, the LAION-2B-en-aesthetic parquet files are available [here](https://huggingface.co/datasets/laion/laion2B-en-aesthetic). To only download a subset, you can create a new parquet with only a subset of records.

To download the Laion chunk and generate embeddings, run the following - 
```
python download_and_generate_embedding.py --parquet-fname {path-to-parquet} \
                               --dump-path {path-to-store-embeddings}
```

### Generate embeddings from ImageFolder

To generate embeddings from ImageFolder (such as generated images), use the same script but specify image folder instead.
```
python download_and_generate_embedding.py --image-folder {path-to-imagefolder} \
                        --dump-path {path-to-store-embeddings}
```

### Embedding similarity search

To perform embedding similarity search run the following - 
```
python similarity_search.py --laion-embedding-folder {path-to-folder} \
                            --generation-embedding-path {path-to-gen-images} \
                            --dump-path {result-dump} \
                            --chunks {chunks}
```
Similarity search performs a large matrix multiplication across embeddings from generated images and LAION images. 

The argument `--laion-embedding-folder` expects a folder that consists of one or more subfolder each containing a single `embeddings.pkl` file. 

The argument `--generation-embedding-path` excepts a path to single `embeddings.pkl` file. 

Since the matrix multiplication can be very large, and cause CUDA OOM we perform it in chunks, the max chunk size is controlled by argument `--chunks`. Increase it for better throughput, but increasing too much may cause OOM.

The resulting file consists of parquet indexes of LAION top-matches for generated images, with similarity score. 