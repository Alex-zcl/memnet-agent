# Публикация `memnet-agent` в PyPI от вашего профиля

На 4 августа 2026 года страница `https://pypi.org/project/memnet-agent/` возвращала 404, то есть проект с таким именем не был опубликован. Это не резервирует имя: проверьте его повторно непосредственно перед первым релизом.

## Что уже подготовлено

- современный `pyproject.toml`;
- wheel и source distribution через setuptools;
- MIT License;
- CLI entry point;
- тесты;
- GitHub Actions CI;
- `release.yml` для PyPI Trusted Publishing без постоянного API-токена.

## Рекомендуемый способ: GitHub + PyPI Trusted Publishing

Trusted Publishing использует короткоживущие OIDC credentials. Секрет `PYPI_API_TOKEN` в GitHub не нужен.

### 1. Проверьте метаданные

В `pyproject.toml`:

- измените `authors`, если нужна другая транслитерация имени;
- при необходимости добавьте `project.urls` после создания GitHub-репозитория;
- измените `name`, если `memnet-agent` уже занят;
- перед каждым релизом увеличивайте `version` одновременно в:
  - `pyproject.toml`;
  - `src/memnet_agent/version.py`.

Версию, уже загруженную в PyPI, нельзя перезаписать.

### 2. Локально проверьте пакет

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
python -m build
python -m twine check dist/*
```

Проверьте wheel в отдельном окружении:

```bash
python -m venv .wheel-test
source .wheel-test/bin/activate
python -m pip install dist/memnet_agent-0.1.0-py3-none-any.whl
python -c "from memnet_agent import MemoryAgent; print('import ok')"
```

### 3. Создайте GitHub-репозиторий

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin git@github.com:YOUR_GITHUB_USERNAME/memnet-agent.git
git push -u origin main
```

### 4. Создайте GitHub environments

В репозитории откройте:

`Settings → Environments`

Создайте:

- `pypi` — рекомендуется включить required reviewers/manual approval;
- `testpypi` — approval необязателен.

### 5. Настройте pending publisher в PyPI

Войдите именно в тот PyPI-профиль, который должен владеть проектом.

Откройте account publishing settings и добавьте GitHub pending publisher:

- PyPI project name: `memnet-agent`;
- GitHub owner: ваш username или организация;
- repository: `memnet-agent`;
- workflow filename: `release.yml`;
- environment: `pypi`.

Аналогично создайте publisher в TestPyPI:

- project name: `memnet-agent`;
- workflow filename: `release.yml`;
- environment: `testpypi`.

Важно: pending publisher не резервирует имя до фактической первой публикации.

### 6. Тестовая публикация

В GitHub откройте:

`Actions → Publish package → Run workflow`

Выберите `testpypi`.

После завершения установите пакет из TestPyPI:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  memnet-agent
```

`--extra-index-url` нужен, чтобы зависимости `numpy` и `scikit-learn` брались из основного PyPI.

### 7. Публикация production-релиза

Создайте git tag, совпадающий с версией:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Workflow соберёт архивы и отправит их в PyPI через environment `pypi`. При включённом approval GitHub сначала запросит ваше подтверждение.

После релиза:

```bash
python -m pip install memnet-agent
```

## Альтернатива: ручная публикация API-токеном

Этот вариант проще для первого эксперимента, но для регулярных релизов Trusted Publishing безопаснее.

```bash
rm -rf dist build
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

Когда Twine запросит credentials:

- username: `__token__`;
- password: полный API token, начинающийся с `pypi-`.

Не сохраняйте token в репозитории, shell history, README или `.pypirc` без защищённого secret storage.

## Следующий релиз

1. Обновите код и changelog.
2. Увеличьте версию, например `0.1.0 → 0.1.1`.
3. Запустите тесты и build.
4. Commit и push.
5. Создайте новый tag `v0.1.1`.
6. Подтвердите deployment в environment `pypi`.
