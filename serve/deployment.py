from fastapi import FastAPI
from ray import serve

from serve.pipeline import Pipeline

app = FastAPI()
@serve.deployment
@serve.ingress(app)
class MotionGenerator:
    def __init__(self, config):
        # 初始化模型并载入权重
        self.pipline = Pipeline(config)

    @app.post("/generate")
    async def generate(self, request):
        json_input = await request.json()
        result = await self.pipline.generate_motion(json_input)
        return {"result": result}
