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

I use Docker to get RedisAI up and running. I'm assuming you can get Docker going on your computer. Once you have, just run the start-redis.sh shell script and it'll pull down and run an image containing RedisAI.

    $ ./start-redis

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

Browses to http://localhost:8000/. You may get CORS issues due to localhost restrictions. I got around this with Firefox by installing the [CORS Everywhere extension](https://addons.mozilla.org/en-US/firefox/addon/cors-everywhere/). ONLY USE IT FOR LOCAL STUFF. DO NOT LEAVE IT ON. CORS is there for a reason.
