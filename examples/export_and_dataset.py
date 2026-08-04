from memnet_agent import MemoryAgent, SleepConfig

agent = MemoryAgent(lambda prompt: "answer", sleep=SleepConfig.off())
agent.learn([
    "Atlas uses PostgreSQL and Redis.",
    "Anna owns the Atlas architecture review.",
    "The review is scheduled for 18 August.",
])
agent.export_graph("atlas-memory.zip")
agent.export_graph("atlas-memory.graphml")
agent.generate_training_dataset("atlas-training.jsonl", iterations=3)
agent.close()
