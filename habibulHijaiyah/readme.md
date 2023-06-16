# ML-HabibulQuran

# Deployment to Google Cloud
  * **Prerequisites** 
    <br>
    Here are several points to consider before proceeding:
    * Install or update to the latest version of the **Google Cloud CLI**
    * Set a default region and zone `asia-southeast2`
    * Enable **Google Container Registry API** and **Cloud Run API** 
      <br><br>
  * **Google Cloud Run**
    <br>
The Google Cloud Run is a fully managed serverless compute platform provided by Google Cloud. It allows you to run stateless containers that are automatically scaled based on incoming requests or events. Cloud Run abstracts away the underlying infrastructure, enabling you to focus on building and deploying your applications without worrying about managing servers or scaling infrastructure.
    <br><br>
    To deploy a machine learning model through Google Cloud Run,
    * Activate **Cloud Shell**
    * Clone ML-HabibulQuran git repository
      ````
      git clone -b ??? "https://github.com/Bangkit-2023-Habibul-Quran/ML-HabibulQuran.git" habibulAyat
      ````
    * Go to the  folder
      ````
      cd habibulAyat
      ````
    * Make sure that you have a `Dockerfile` file and requirements.txt using the `ls` command on **Cloud Shell**
    * Example Dockerfile
      ````
      FROM python:3.10.11
      # Use the official lightweight Python image.
      # https://hub.docker.com/_/python
      # FROM python:3.9-slim
      # Allow statements and log messages to immediately appear in the Knative logs
      ENV PYTHONUNBUFFERED True
      # Copy local code to the container image.
      ENV APP_HOME /app
      WORKDIR $APP_HOME
      COPY . ./
      # Install production dependencies.
      RUN pip install -r requirements.txt
      # Set Gunicorn configuration
      ENV GUNICORN_CMD_ARGS="--timeout 604800???"
      # Run the web service on container startup. Here we use the gunicorn
      # webserver, with one worker process and 8 threads.
      # For environments with multiple CPU cores, increase the number of workers
      # to be equal to the cores available.
      # Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
      CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
      ````
   * Creating Docker image
      ```
      gcloud builds submit --tag gcr.io/habibulquran/habibulayat
      ```
   * Deploy image to Cloud Run
     * On GCP console, go to **Navigation Menu -> Cloud Run**
     * Click **Create Service** 
     * Choose Deploy one revision from an existing container image
       * Select Container image URL -> Container Registry -> Your image -> Click Select
     * Set region to `asia-southeast2`
     * Set maximum number of instances to 5
     * Select Allow unauthenticated invocations
     * Click dropdown on Container, Networking, Security
       * Set memory to 2 GB memory 1vCPU
       * Set request time out to 3600
     * Leave the rest of the menu to default
     * Click Create (this may take a few minutes)
