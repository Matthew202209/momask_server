from fastapi import FastAPI
from ray import serve

from serve.pipeline import Pipeline

app = FastAPI()
@serve.deployment(
    # 核心配置：自动扩缩容
    autoscaling_config={
        "min_replicas": 1,              # 需求小时，最少保持 1 个副本
        "max_replicas": 10,              # 需求大时，最多扩容到 5 个副本
        "target_ongoing_requests": 10,    # 每个副本平均处理 2 个并发请求，超过就扩容
    },
    ray_actor_options={"num_cpus": 0.5, "num_gpus": 0.3} # 每个副本占用 0.5 个 CPU
)
@serve.ingress(app)
class MotionGenerator:
    def __init__(self):
        # 初始化模型并载入权重
        self.pipline = Pipeline(model_path)

    @app.post("/generate")
    async def generate(self, request):
        json_input = await request.json()
        result = await self.pipline.generate_motion(json_input)
        return {"result": result}
