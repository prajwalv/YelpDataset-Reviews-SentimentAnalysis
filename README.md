# YelpDataset-Reviews-SentimentAnalysi
Semester project for Data Mining(CMPE255). In this project, I have performed sentiment analysis on Yelp, Reviews dataset.
All of our dataset samples can be found under dataset folder.

## extract.py
We had tar file as our dataset. This script will extract the tar files.
To run this script we have to change the path of the tar file.

##json_to_csv_convertor.py
When we extracted the files, we had json files. We used this script to convert the jsom to csv files.
To run this script we have to pass the path of the json file as a parameter.

## classifyingRestaurantCategoryfromBusiness.py
yelp_academic_dataset_business.csv file had different business categories. As we decided to use only restaurant dataset we generated a new onlyRestaurants.csv file with business_id and categories only.
To run this script we have to pass the correct path of yelp_academic_dataset_business.csv file.

## getRestrauntReviewOnly.py
yelp_academic_dataset_review.csv file has around 6000000 reviews from all the business. As we wanted only restaurant reviews we generated a new FilteredRestaurantReviews.csv file with business_id, reviews and star.
generated a new onlyRestaurants.csv file with business_id and categories only.
To run this script we have to pass the correct path of yelp_academic_dataset_review.csv file.

## predictNegativeOrPositiveReview.py
This script takes reviews from FilteredRestaurantReviews.csv file and classifies them as a positive or a negative review.
To run this script we have to pass the correct path of FilteredRestaurantReviews.csv file.

