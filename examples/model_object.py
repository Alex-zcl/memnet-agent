from memnet_agent import MemoryAgent


class FrameworkModel:
    def invoke(self, prompt: str):
        return {"content": "Ответ framework-модели"}


agent = MemoryAgent(model=FrameworkModel())
print(agent.ask("Привет"))
agent.close()
