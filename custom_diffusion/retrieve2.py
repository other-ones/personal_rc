from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality

IMAGE_BASE_URL = "https://github.com/rom1504/clip-retrieval/raw/main/tests/test_clip_inference/test_images/"

def log_result(result):
    id, caption, url, similarity = result["id"], result["caption"], result["url"], result["similarity"]
    print(f"id: {id}")
    print(f"caption: {caption}")
    print(f"url: {url}")
    print(f"similarity: {similarity}")
    # display(Image(url=url, unconfined=True))

client = ClipClient(
    url="https://knn.laion.ai/knn-service",
    indice_name="laion5B-L-14",
    aesthetic_score=9,
    aesthetic_weight=0.5,
    modality=Modality.IMAGE,
    num_images=10,
)

cat_results = client.query(text="an image of a cat")
print(cat_results,'cat_results')
log_result(cat_results[0])