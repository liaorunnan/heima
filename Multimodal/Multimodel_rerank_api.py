

from fastapi import FastAPI, Body
from modelscope import AutoModel
import uvicorn

app = FastAPI()


model = AutoModel.from_pretrained(
        '/root/autodl-tmp/.cache/modelscope/hub/models/jinaai/jina-reranker-m0',
        dtype="auto",
        trust_remote_code=True,
    )

model.to('cuda')  # or 'cpu' if no GPU is available
model.eval()

@app.post("/rank")
def rank(text_1: str = Body(..., embed=True),
    text_2: str = Body(..., embed=True),
    query_type: str = Body("text", embed=True),
    answer_type: str = Body("text", embed=True)):
    # comment out the flash_attention_2 line if you don't have a compatible GPU

    print(text_1)
    

    image_pairs = [[text_1, text_2]]
    scores = model.compute_score(image_pairs, max_length=2048, query_type=query_type, doc_type=answer_type)
    return scores
        
if __name__ == '__main__':
    #启动服务
    uvicorn.run(app, host="0.0.0.0", port=6006)