#### Reference
* The BERT saved model was generated using code adapted from
https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

#### Downloading the saved model
* The saved model directory can be downloaded from https://fiddler-data.s3-us-west-1.amazonaws.com/imdb_bert/
* We suggest using aws cli for the download, using the command
aws s3 sync s3://fiddler-data/imdb_bert/saved_model <your 
path>/onebox/fiddler/repo/onebox/imdb_rnn/imdb_bert/saved_model

#### Model settings
* The max_allowed_error in the get_model function in package.py is set to 5,
 which means that IG will continue to run till the error is less than 5%.
* In general, increasing the error bound will make attributions less reliable
and faster, and vice-versa
