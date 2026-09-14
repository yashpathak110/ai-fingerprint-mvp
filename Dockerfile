FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Ensure python-multipart is in requirements before deploying!
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# We can no longer EXPOSE a static port as Render assigns one dynamically.
# Render automatically handles port exposure based on the assigned PORT variable.

# Switch CMD to run the python file directly, enabling the port logic we added in Step 1.
CMD ["python", "server.py"]