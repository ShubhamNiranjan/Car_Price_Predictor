This project is based upon the dataset provided on kaggle ( Containing approx 4000 rows). 
The dataset had a number of columns as km_driven, seller_type, transmission and other information which are correlated to selling_price (as this is a supervised learning task).
First I removed the duplicate values then performed label encoding on some selected columns but performed target encoding on brand or name column as the number of entries are of wide-range.
Then trained the RandomForestRegressor model on the dataset, and used the same to predict the value on new query.
Used streamlit for the frontend and some part of html for color and hover effects on st.button, while predicting used st.snow() as an animation.
