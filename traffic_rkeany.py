# App to predict the chances of admission using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


# Reading the pickle file that we created before 
model_pickle = open('reg_traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('Input_format.csv')
default_df.drop(columns=["traffic_volume"], inplace=True)

st.sidebar.image("traffic_sidebar.jpg", use_column_width = True, caption="Traffic Volume Predictor")

# Create a sidebar for input collection
st.sidebar.header('Input Features')
st.sidebar.write("You can either upload your data file or manually enter traffic volume features")

with st.sidebar.expander("Option 1: Upload a CSV File"):
    st.write("Upload a CSV file containing traffic details.")
    file = st.file_uploader("Choose a CSV File", type='csv', accept_multiple_files=False)
    st.header("Sample Data format for Upload")
    st.dataframe(default_df.head())
    st.write("Ensure your uploaded file has the same column names and data types as shown above")

with st.sidebar.expander("Option 2: Fill Out Form"):
    st.write("Enter the diamond details manually using the form below.")
    with st.form("Form Data"):

        # Input for 'holiday' (categorical)
        holiday_options = default_df['holiday'].unique().tolist()
        holiday = st.selectbox('Choose if the day falls on a holiday', options=holiday_options)

        # Input for 'weather_main' (categorical)
        weather_main_options = default_df['weather_main'].unique().tolist()
        weather_main = st.selectbox('State what the weather looks like', options=weather_main_options)

        # Input for 'month' (categorical)
        month_options = sorted(default_df['month'].unique().tolist())
        month = st.selectbox('Choose the month', options=month_options)

        # Input for 'weekday' (categorical)
        weekday = st.selectbox('Choose the day of the week', 
                               options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        # Input for 'hour' (categorical)
        hour_options = sorted(default_df['hour'].unique().tolist())
        hour = st.selectbox('Choose the hour of the day (military time)', options=hour_options)

        # Input for 'temp' (numerical)
        min_temp = float(default_df['temp'].min())
        max_temp = float(default_df['temp'].max())
        temp = st.number_input('What is the temperature', min_value=min_temp, max_value=max_temp, value=min_temp, step=0.01)

        # Input for 'rain_1h' (numerical)
        min_rain_1h = float(default_df['rain_1h'].min())
        max_rain_1h = float(default_df['rain_1h'].max())
        rain_1h = st.number_input('Rain (1h)', min_value=min_rain_1h, max_value=max_rain_1h, value=min_rain_1h, step=0.1)

        # Input for 'snow_1h' (numerical)
        min_snow_1h = float(default_df['snow_1h'].min())
        max_snow_1h = float(default_df['snow_1h'].max())
        snow_1h = st.number_input('Snow (1h)', min_value=min_snow_1h, max_value=max_snow_1h, value=min_snow_1h, step=0.1)

        # Input for 'clouds_all' (numerical)
        min_clouds_all = int(default_df['clouds_all'].min())
        max_clouds_all = int(default_df['clouds_all'].max())
        clouds_all = st.number_input('Clouds (All)', min_value=min_clouds_all, max_value=max_clouds_all, value=min_clouds_all, step=1)
        
        submit_button = st.form_submit_button("Submit Form Data")


# Set up the app title and image
st.title('Traffic Volume Predictor')
st.write("This app helps you estimate the traffic volume based on selected features.")
st.image('traffic_image.gif', use_column_width = True)

if submit_button:
    st.success("Form data submitted successfully.")
elif file:
    st.success("CSV file uploaded successfully.")
else:
    st.info("Please choose a data input method to proceed.")


# Get the prediction with its intervals
st.write("Select alpha value for prediction intervals")
alpha = st.slider("Select alpha value for prediction intervals", min_value= 0.01, max_value=0.50, step=0.01)

expected_columns = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'holiday_Columbus Day', 'holiday_Independence Day', 
                    'holiday_Labor Day', 'holiday_Martin Luther King Jr Day', 'holiday_Memorial Day', 'holiday_New Years Day', 
                    'holiday_None', 'holiday_State Fair', 'holiday_Thanksgiving Day', 'holiday_Veterans Day', 
                    'holiday_Washingtons Birthday', 'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog', 
                    'weather_main_Haze', 'weather_main_Mist', 'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow', 
                    'weather_main_Squall', 'weather_main_Thunderstorm', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 
                    'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'weekday_Monday', 'weekday_Saturday', 
                    'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday', 'hour_1', 'hour_2', 'hour_3', 
                    'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 
                    'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23']



if file:
    st.subheader(f"Prediction Results with {(1-alpha)*100:.0f}% Confidence Interval")
    df1 = pd.read_csv(file)
    
    df = pd.get_dummies(df1, columns=['holiday', 'weather_main', "month", "weekday", "hour"], drop_first=True)
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[expected_columns]

    # Make predictions and get the intervals
    predictions, intervals = reg_model.predict(X, alpha=alpha)

    # Round predictions and intervals to 2 decimal points
    df["Predicted Volume"] = np.round(predictions, 2)
    df["Lower Prediction Limit"] = np.round(np.maximum(0, intervals[:, 0]), 2)
    df["Upper Prediction Limit"] = np.round(intervals[:, 1], 2)

    # Add results to the original DataFrame
    df1["Predicted Volume"] = round(df["Predicted Volume"], 1)
    df1["Lower Prediction Limit"] = round(df["Lower Prediction Limit"], 1)
    df1["Upper Prediction Limit"] = round(df["Upper Prediction Limit"], 1)

    # Display the dataframes
    st.dataframe(data=df1)

else:
    # Encode the inputs for model prediction
    encode_df = default_df.copy()

    # Combine the list of user data as a row to default_df
    # Add a new row to encode_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df, columns=['holiday', 'weather_main', "month", "weekday", "hour"], drop_first=True)

    encode_dummy_df = encode_dummy_df.reindex(columns=expected_columns, fill_value=0)


    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)


    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1]


    # Ensure limits are within [0, inf]
    pred_value = max(0, pred_value)
    lower_limit = max(0, lower_limit[0][0])
    upper_limit = max(lower_limit, upper_limit[0][0])

    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")

    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value:.0f}")
    st.write(f"**Confidence Interval** ({(1-alpha)*100:.0f}%): [{lower_limit:.2f}, {upper_limit:.2f}]")


# Additional tabs for DT model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals",
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")

