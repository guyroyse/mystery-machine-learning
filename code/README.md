# Mystery Machine Learning: Code Demo

This codebase is a small application that classifies lines from Scooby Doo and will tell you which member of Mystery Incorporated said it. It is made up of four parts. The Python script to build the model, RedisAI to host the model, a Node and Express service that talks to Redis and makes inferences, and a web client to consume the service.

## Building the Model

Setup a virtual environment for Python:

    $ python3 -m venv venv

Activate that environment:

    $ source venv/bin/activate

Install dependencies:

    $ pip install -r requirements.txt

Build the model:

    $ python build-model.py

If you need to deactivate the virtual environment:

    $ deactivate

## Up and Running with RedisAI

I use Docker and Docker Compose to get RedisAI up and running. I'm assuming you can get Docker going on your computer. Once you have, just run the following command:

    $ cd docker
    $ docker compose up

And that's it.

## Running the Service

Install all the node dependencies:

    $ npm install

Run the serivce:

    $ npm start

## Running the Client

Make sure you Python environment is activated. We're gonna use a simple web server from it:

    $ source venv/bin/activate

Go into the web app:

    $ cd web-client

Start the Python server:

    $ python -m http.server

Browse to http://localhost:8000/.
