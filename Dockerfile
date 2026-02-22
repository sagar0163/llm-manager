FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-c", "from llm_manager import LLMManager; import asyncio; asyncio.run(LLMManager().initialize())"]
