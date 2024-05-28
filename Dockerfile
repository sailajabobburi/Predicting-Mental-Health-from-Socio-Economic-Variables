#use the official python image
FROM python:3.11-slim

# set the working directory in the container
WORKDIR /src

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . .

#expose the port the app runs on
EXPOSE 8000

#set the python path
ENV PYTHONPATH=/src

# command to run on container start
CMD [ "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000" ]