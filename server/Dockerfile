FROM arm64v8/python:3.9-slim-buster  

# Working directory
WORKDIR /app

# Copy dependencies
COPY ./requirements.txt ./
RUN pip install -r requirements.txt

# Copy your server code
COPY *.py ./

# Expose the port used by your FastAPI app (e.g., 8000)
EXPOSE 8000

# Command to start the server 
CMD ["uvicorn", "detection_server:app", "--host", "0.0.0.0", "--port", "8000"]