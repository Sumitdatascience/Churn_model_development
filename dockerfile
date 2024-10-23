# Step 1: Use an official Python runtime as a base image
FROM python:3.10

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the project files into the container
COPY flask_app/ /app/

# Step 4: Install the required Python dependencies
RUN pip install -r requirements.txt

# Step 5: Expose a port if your project needs it (e.g., for an API or web service)
EXPOSE 5000

# Step 6: Run the application (replace "app.py" with your app's entry point)
# CMD ["python", "app.py"]

CMD ["gunicorn","-b","0.0.0.0:5000","app:app"]
