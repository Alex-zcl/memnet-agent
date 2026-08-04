from memnet_agent import MemoryAgent


def demo_model(prompt: str) -> str:
    # Replace with your real model call.
    if "Friday" in prompt or "пятниц" in prompt.lower():
        return "Релиз запланирован на пятницу."
    return "Информация сохранена."


with MemoryAgent(model=demo_model, storage_path="demo-memory.sqlite") as agent:
    agent.learn("Релиз проекта Atlas запланирован на пятницу.", source="release-note")
    print(agent.ask("Когда релиз Atlas?"))
