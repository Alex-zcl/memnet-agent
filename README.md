# memnet-agent

`memnet-agent` превращает любую текстовую модель в агента с долговременной ассоциативной памятью-графом.

Модель отвечает за генерацию текста. Библиотека отвечает за:

- поиск связанных воспоминаний перед инференсом;
- добавление новых запросов и ответов в граф;
- чтение документов без генерации ответа;
- decay, merge, split, prune и ассоциативный synthesis во время «сна»;
- импорт, сохранение и перенос графа;
- генерацию итеративного train-датасета из текущей памяти.

> Статус: alpha. Синтезированные узлы являются ассоциациями, а не автоматически проверенными фактами.

## Установка

После публикации в PyPI:

```bash
pip install memnet-agent
```

Локально из исходников:

```bash
python -m pip install -e ".[dev]"
```

## 1. Самый простой агент

Передайте callable, который получает строку prompt и возвращает строку ответа:

```python
from memnet_agent import MemoryAgent


def my_model(prompt: str) -> str:
    # Здесь может быть локальная нейросеть, HTTP-клиент или SDK провайдера.
    return "Ответ модели"


agent = MemoryAgent(model=my_model)
print(agent.ask("Запомни, что релиз запланирован на пятницу"))
print(agent.ask("Когда запланирован релиз?"))
agent.close()
```

`agent(text)` является коротким алиасом для `agent.ask(text)`.

## 2. Передача готового объекта модели

Автоматически поддерживаются объекты с методом:

- `generate(prompt)`;
- `invoke(prompt)`;
- `predict(prompt)`;
- `complete(prompt)`;
- `chat(prompt)`.

```python
from memnet_agent import MemoryAgent

llm = SomeFrameworkModel(...)
agent = MemoryAgent(model=llm)
answer = agent.ask("Что мы обсуждали раньше?")
```

Если метод называется иначе:

```python
agent = MemoryAgent(model=llm, model_method="run_text")
```

Ответы форматов `str`, LangChain-подобный `.content`, OpenAI-подобный `choices`, словари и типовые результаты Hugging Face pipeline преобразуются в текст автоматически. Для нестандартного SDK используйте маленький callable-адаптер.

## 3. Как проходит инференс

При `agent.ask(text)` библиотека:

1. ищет релевантные узлы по TF-IDF, тегам, strength и confidence;
2. добавляет найденный контекст в prompt;
3. вызывает переданную модель;
4. возвращает строку ответа;
5. по умолчанию сохраняет запрос и ответ в граф;
6. связывает новый запрос с использованными воспоминаниями.

Для отладки retrieval:

```python
result = agent.ask_with_trace("Что известно о договоре?")

print(result.answer)
for hit in result.memories:
    print(hit.score, hit.node.text)
```

Отключение обновления графа:

```python
agent = MemoryAgent(model=my_model, update_memory=False)
```

Сохранять запросы, но не ответы:

```python
agent = MemoryAgent(model=my_model, remember_responses=False)
```

## 4. Чтение информации без ответа

`learn()` и `ingest()` добавляют материал в граф, но ничего не отвечают пользователю:

```python
agent.learn(
    """
    Проект Atlas использует PostgreSQL.
    Технический владелец — Анна.
    Следующий review назначен на 18 августа.
    """,
    source="project-atlas-notes",
)

agent.learn("docs/architecture.txt")
```

Большой текст автоматически разбивается на перекрывающиеся смысловые chunks.

Можно скрытно прогнать каждый chunk через основную модель перед сохранением:

```python
agent.learn(document, preprocess_with_model=True)
```

В этом режиме модель получает инструкцию создать компактные фактические notes; результат пользователю не выводится.

## 5. Режимы сна

### A. Сон после простоя

Это режим по умолчанию. После периода без активных запросов выполняется обслуживание графа:

```python
from memnet_agent import MemoryAgent, SleepConfig

agent = MemoryAgent(
    model=my_model,
    sleep=SleepConfig.idle(
        after_seconds=20,
        maintenance_every_seconds=60,
    ),
)
```

### B. Сон по часам

В указанное окно обычный инференс блокируется, но `learn()` продолжает работать, а граф обслуживается в фоне:

```python
agent = MemoryAgent(
    model=my_model,
    sleep=SleepConfig.scheduled(
        start="02:00",
        end="05:00",
        timezone="Europe/Amsterdam",
    ),
)
```

Разрешить срочные запросы во время окна сна:

```python
sleep=SleepConfig.scheduled(
    start="02:00",
    end="05:00",
    timezone="Europe/Amsterdam",
    allow_interrupt=True,
)
```

### C. Постоянные memory-workers

Несколько daemon workers регулярно консолидируют общий граф, пока основной агент отвечает:

```python
agent = MemoryAgent(
    model=my_model,
    sleep=SleepConfig.workers(
        count=3,
        maintenance_every_seconds=15,
    ),
)
```

Workers не генерируют пользовательские ответы. Операции над одним графом синхронизированы, чтобы не повреждать рёбра и индексы.

### Ручной сон

```python
agent = MemoryAgent(model=my_model, sleep=SleepConfig.off())
report = agent.sleep(max_syntheses=5)
print(report)
```

## 6. Сохранение и экспорт графа

```python
agent.export_graph("memory.sqlite")  # основной рабочий формат
agent.export_graph("memory.json")    # переносимый JSON
agent.export_graph("memory.graphml") # Gephi / Cytoscape / NetworkX
agent.export_graph("memory.zip")     # SQLite + JSON + manifest
```

Автосохранение после изменений:

```python
agent = MemoryAgent(
    model=my_model,
    storage_path="state/memory.sqlite",
    auto_save=True,
)
```

Рекомендуется закрывать агент через context manager:

```python
with MemoryAgent(model=my_model, storage_path="memory.sqlite") as agent:
    print(agent.ask("Привет"))
```

## 7. Подключение внешнего графа

При создании:

```python
agent = MemoryAgent.from_graph(my_model, "shared-memory.zip")
```

Или позже:

```python
agent.load_graph("team-memory.json", replace=True)
```

Объединение внешнего графа с текущим:

```python
agent.load_graph("second-memory.sqlite", replace=False)
```

Импортируются SQLite, JSON и zip bundles. GraphML предназначен для визуализации и экспорта, а не для обратного импорта.

## 8. Итеративный train-датасет

```python
agent.generate_training_dataset(
    "training.jsonl",
    iterations=3,
    max_examples_per_iteration=5000,
    format="chat",  # chat | instruction | graph
)
```

Чтобы между итерациями граф проходил консолидацию:

```python
agent.generate_training_dataset(
    "training-evolving.jsonl",
    iterations=4,
    consolidate_between_iterations=True,
)
```

Потоковая работа без файла:

```python
for record in agent.iter_training_dataset(iterations=2):
    train(record)
```

## 9. Асинхронная модель

```python
answer = await agent.aask("Сформируй краткое резюме")
result = await agent.aask_with_trace("Что связано с Atlas?")
```

## 10. CLI

```bash
memnet-agent info memory.sqlite
memnet-agent validate memory.sqlite
memnet-agent convert memory.sqlite memory.graphml
memnet-agent dataset memory.sqlite train.jsonl --iterations 3 --format chat
```

## Публичный API

```python
from memnet_agent import (
    AgentResult,
    AssociativeMemory,
    MemoryAgent,
    MemoryHit,
    SleepConfig,
)
```

Низкоуровневый `MemoryNet` оставлен доступным для прямой работы с графом.

## Ограничения версии 0.1.0

- Retrieval пересчитывает TF-IDF по графу и рассчитан на небольшие и средние локальные памяти. Для миллионов узлов нужен embedding/ANN backend.
- Background workers используют потоки и один синхронизированный граф; это безопасный maintenance, но не распределённая обработка.
- Библиотека не проверяет истинность текста модели.
- Не храните секреты и персональные данные без собственного шифрования и политики доступа.

## Разработка

```bash
python -m pip install -e ".[dev]"
pytest
python -m build
python -m twine check dist/*
```

Пошаговая публикация находится в [`PUBLISHING.md`](PUBLISHING.md).

## Лицензия

MIT.
