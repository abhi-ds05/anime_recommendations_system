# Anime Recommendation System🎌

[![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?logo=plotly&logoColor=white)
[![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Keras Tuner](https://img.shields.io/badge/-Keras%20Tuner-FF6F00?logo=keras&logoColor=white)](https://keras-team.github.io/keras-tuner/)
[![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white)

## Table of Contents

- [Demo](#demo)
- [Overview](#overview-)
- [Code Walkthrough](#code-walkthrough)
- [About the Dataset](#about-the-dataset)
- [Motivation](#motivation)
- [Acknowledgments](#acknowledgments)
- [Installation](#installation)
- [Directory Tree](#directory-tree)
- [Bug / Feature Request](#bug--feature-request)
- [Future scope of project](#future-scope)


## Demo 🎥

https://github.com/abhi-ds05/anime-recommendation-system/assets/raw/main/resource/demo-video.mp4

Or click to watch below:

<video src="resource/demo-video.mp4" controls autoplay loop width="100%"></video>


Note: If the website link provided above is not working, it might mean that the deployment has been stopped or there are technical issues. We apologize for any inconvenience.

- Please consider giving a ⭐ to the repository if you find this app useful.
- A quick preview of my anime recommendation system.

## Overview 🌟📚

Welcome to the Anime Recommendation System! This project aims to provide personalized anime recommendations based on collaborative filtering techniques.

The application utilizes user-based collaborative filtering to find similar users based on their anime preferences and recommends animes liked by similar users that the target user has not watched yet. Additionally, the system employs item-based collaborative filtering to find similar animes based on their features (e.g., genres, synopsis) and recommends animes similar to the one provided by the user.

The dataset used for training and recommendation includes various anime titles, user ratings, and anime features such as genres and synopses. The model was trained using TensorFlow and Keras to create anime embeddings for both users and animes, facilitating efficient similarity comparisons.

Feel free to explore and enjoy the exciting world of anime recommendations with our innovative system!

## Code Walkthrough

Welcome to the Code Walkthrough section of the Anime Recommendation System project! This project is divided into two notebooks, which can be found in the `notebooks` folder of this repository. Let's dive into the two notebooks that make up this awesome project:

#### Notebook 1: `anime-recommendation-1.ipynb`

In Notebook 1, we embark on a journey into the world of anime data analysis. The main objectives of this notebook are as follows:

- **Understand the dataset**: We take a closer look at the dataset, examining its structure and contents to get familiar with the data.
- **Perform data preprocessing**: We clean and prepare the data for analysis, ensuring that it's in a suitable format for our recommendation models.
- **Interactive Data Visualization**: To gain valuable insights, we utilize the power of [Plotly](https://plotly.com/), a fantastic library that provides us with interactive and engaging visualizations.

#### Notebook 2: `anime-recommendation-2.ipynb`

In Notebook 2, we take the next step in our journey by training our recommendation model.

- Part 1: Collaborative Filtering

  Here, we delve into collaborative filtering, a popular recommendation technique that suggests animes to users based on the preferences of similar users or similar animes.

  My key steps were:

  1. **Data Preprocessing**: I loaded the datasets, perform data scaling, and encode user and anime IDs to prepare the data for model training.
  2. **Model Architecture**: To facilitate collaborative filtering, I created a neural network-based model. The model uses embeddings to represent users and animes in a lower-dimensional space, capturing their underlying preferences.
  3. **Model Training**: Using TensorFlow, I trained the collaborative filtering model to predict user ratings for animes. The optimization process ensures that the model learns to recognize patterns and make accurate recommendations.
  4. **Recommendation Generation**: With the trained model, we can now find similar animes and users 😎.

- Part 2: Content-Based Filtering

  The second part of this notebook explores content-based filtering, an alternate recommendation technique. Content-based filtering suggests animes to users based on attributes such as genres and ratings. Here, my key steps were:

  1. **TF-IDF Vectorization**: I created a TF-IDF matrix for anime genres to quantify the importance of genres in each anime's description.
  2. **Cosine Similarity**: By computing cosine similarity between animes based on their genre descriptions, we can determine their similarity.
  3. **Content-Based Recommendation**: Leveraging the computed similarity scores and ratings, we now can recommend animes that are similar to a given anime, considering their genre and score.

  We've got an exciting mix of collaborative and content-based filtering models, ensuring we can deliver diverse and accurate anime recommendations tailored to the preferences of each user. 🤗

  Happy anime recommending! 🎊


## About the Dataset

The dataset used in the Anime Recommendation System project offers a wealth of valuable information, encompassing anime characteristics, ratings, popularity, viewership, user behavior, and preferences. It serves as a comprehensive resource for conducting diverse analyses, such as identifying top-rated anime, exploring popular genres, and gaining insights into viewer trends. With this dataset, personalized recommendation systems can be developed, user behavior can be analyzed, and clustering can be employed to understand anime trends and user preferences. Additionally, the dataset enables examination of user interactions, ratings, and engagement with anime, providing essential inputs for collaborative filtering and similarity analysis.



## Motivation

🎬 As a passionate anime enthusiast, I've always been captivated by the rich storytelling, vibrant characters, and imaginative worlds of anime. Every series I watched sparked a sense of wonder and emotional connection that stayed with me. This love for anime, combined with my curiosity for technology, inspired me to take it a step further — to bring the joy of discovering great anime to others through intelligent recommendations.



## Installation

This project is written in Python 3.11.4. If you don't have Python installed, you can download it from the [official website](https://www.python.org/downloads/). If you have an older version of Python, you can upgrade it using the pip package manager, which should be already installed if you have Python 2 >=2.7.9 or Python 3 >=3.4 on your system.
To install the required packages and libraries, you can use pip and the provided requirements.txt file. First, clone this repository to your local machine using the following command:

```

Once you have cloned the repository, navigate to the project directory and run the following command in your terminal or command prompt:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages and libraries needed to run the project.

If you prefer, you can also create a virtual environment to manage the project dependencies separately. This helps in isolating the project's environment from the system-wide Python installation.

## Directory Tree

```
|   .gitignore
|   app.py
|   LICENSE.md
|   package-lock.json
|   package.json
|   README.md
|   requirements.txt
|
+---model
|       anime-dataset-2023.pkl
|       anime_encoder.pkl
|       myanimeweights.h5
|       users-score-2023.csv
|       user_encoder.pkl
|
+---notebooks
|       anime-recommendation-1.ipynb
|       anime-recommendation-2.ipynb
|
+---resource
|       anime.mp4
|
+---static
|       main.css
|       style.css
|
\---templates
        index.html
        recommendations.html
```

## Bug / Feature Request

If you encounter any bugs or issues with the anime recommendation app, please let me know by opening an issue on my [GitHub repository](https://github.com/abhi-ds05/anime_recommendations_system/issues). Be sure to include the details of your query and the expected results. Your feedback is valuable in helping me improve the app for all users. Thank you for your support!

## Future Scope

1. **Implement Hybrid Recommendation System**: Combine collaborative filtering and content-based filtering techniques to create a hybrid recommendation system.
2. **Include User Feedback and Reviews**: Incorporate user feedback and reviews into the recommendation system to improve the accuracy of recommendations.
3. **Explore Deep Learning Models**: Experiment with advanced deep learning models, such as RNNs and transformer-based architectures, to enhance recommendation performance.
4. **Real-Time Recommendation Updates**: Implement a real-time recommendation system that continuously updates suggestions based on users' interactions.
5. **Integrate External Data Sources**: Consider integrating external data sources, such as user demographics and anime-related news, to personalize recommendations.
6. **Anime Sentiment Analysis**: Perform sentiment analysis on anime reviews to gauge audience sentiments towards specific animes.
7. **User Clustering**: Cluster users based on preferences to provide better personalized recommendations and targeted marketing strategies.
8. **Interactive Web Interface**: Develop a user-friendly web interface for exploring recommendations and detailed anime information.
9. **Social Media Integration**: Allow users to share favorite animes and recommendations on social media platforms.
10. **Anime Popularity Trend Analysis**: Conduct time series analysis to identify trends in anime popularity over different seasons and years.
11. **Personalized Watchlists**: Create personalized watchlists for users, curating a list of animes based on their preferences.
12. **Sentiment-Based Filtering**: Implement sentiment-based filtering for recommending animes based on users' emotions.
