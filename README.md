# qranalytics-nlp
The QRAnalytics-NLP project is a Text classifying API for the [QuikReply-Bot framework](https://github.com/andrewszucs/quikreply-bot). Deploying the API and adding it to the chatbot instance will enable 2 functionalities in the Analytics module:
1. Extracting the 6 keywords from every conversation that happens with your bot. Useful for classifying the conversation based on the chatbots functionality.
2. Sentiment analysis for every message.

## Installation
1. Clone the repository to your computer.
2. Use the ```.ipynb``` Jupyter Notebook files to create a classifying model for both functionalities.
   - The sentiment analysis should work out of the box.
3. Deploy the Flask Python server and specify the APIs URL in the chatbot's ```.env``` file.
   - For Heroku you're going to need to use a buildpack:
   ```heroku create <name of the app> --buildpack https://github.com/kennethreitz/conda-buildpack.git```