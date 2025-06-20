Sentiment Analyis for English text
===========================

1. With terminal navigate to the root of this repository
--------------------------------------------------------

2. Build docker image
---------------------
.. code-block::

    docker build -t image_name .

3. Run container
----------------
.. code-block::

    docker run --name container_name -p 8000:8000 image_name

4. Output will contain
----------------------
INFO:     Uvicorn running on http://127.0.0.1:8000

Use this url in chrome to see the model frontend;
use http://127.0.0.1:8000/docs for testing the model in the web interface.

5. Query model
--------------
    
 #. Via web interface (chrome):
        http://127.0.0.1:8000/docs -> test model
