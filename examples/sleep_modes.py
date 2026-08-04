from memnet_agent import MemoryAgent, SleepConfig

model = lambda prompt: "ok"

idle_agent = MemoryAgent(model, sleep=SleepConfig.idle(after_seconds=30))
night_agent = MemoryAgent(
    model,
    sleep=SleepConfig.scheduled(
        start="02:00",
        end="05:00",
        timezone="Europe/Amsterdam",
    ),
)
worker_agent = MemoryAgent(
    model,
    sleep=SleepConfig.workers(count=3, maintenance_every_seconds=20),
)

idle_agent.close()
night_agent.close()
worker_agent.close()
