import numpy as np
import pandas as pd
import os


# data fetching 
df_unprocessed = pd.read_csv('./data/raw/df.csv')

# data preprocessing steps
def data_preprocessed(df):

    categories_to_replace=["Payment by installment (Hire Purchase by Roger's Capital)","Payment by installment (Hire Purchase by CIM Finance)"]

    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(categories_to_replace, 'Installment')

    categories_to_replace=['Mips Juice',"Mips Bank Transfer",'Mips POP']

    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(categories_to_replace, 'Mips')

    # List of categories to be replaced with 'Electronics'
    categories_to_replace = [
    'Mobile', 'Smartphones', 'Computing', 'Wearables', 'TV & Audio', 
    'Television', 'Televisions', 'Portable Speaker', 'Computer Speakers', 
    'High Pressure Cleaners', 'Laptops', 'Desktop PCs, Laptops & Notebooks', 
    'Networking', 'Monitor', 'Mobile Phone Accessories', 'Tablets and Accessories', 
    'Headset', 'Keyboard & Mouse','Desktop Pcs  Laptops & Notebooks','Headphones','All in One Printer','Other Computer Accessories','Desktop Pcs  Laptops & Notebooks '
    ]

    # Replace the specified categories with 'Electronics'
    df['PreferredOrderCat'] = df['PreferredOrderCat'].replace(categories_to_replace, 'Electronics')

    # Display the updated DataFrame to the user
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Category Mapping to Electronics", dataframe=df)

    # List of categories to be replaced with 'Home Appliances'
    categories_to_replace_home_appliances = [
    'Small Appliances', 'Home Appliances', 'Refrigerators', 'Microwaves', 
    'Wet & Dry Vacuum Cleaner', 'Stick Vacuum Cleaners', 'Air Purifiers', 
    'Fryers', 'Cookwares', 'Built-in Ovens', 'Gas Water Heaters', 'Kettles', 
    'Cooker Hoods', 'Hair Dryers', 'Air Conditioner', 'Tools', 
    'Hedge Trimmer', 'Hot Deals', 'Dry Vacuum Cleaners','Food Processor'
    ]

    # Replace the specified categories with 'Home Appliances'
    df['PreferredOrderCat'] = df['PreferredOrderCat'].replace(categories_to_replace_home_appliances, 'Home Appliances')



    # List of categories to be replaced with 'Home Appliances'
    categories_to_replace_home_appliances = [
    'Small Appliances', 'Home Appliances', 'Refrigerators', 'Microwaves', 
    'Wet & Dry Vacuum Cleaner', 'Stick Vacuum Cleaners', 'Air Purifiers', 
    'Fryers', 'Cookwares', 'Built-in Ovens', 'Gas Water Heaters', 'Kettles', 
    'Cooker Hoods', 'Hair Dryers', 'Air Conditioner', 'Tools', 
    'Hedge Trimmer', 'Hot Deals', 'Dry Vacuum Cleaners','Food Processor'
    ]

    # Replace the specified categories with 'Home Appliances'
    df['PreferredOrderCat'] = df['PreferredOrderCat'].replace(categories_to_replace_home_appliances, 'Home Appliances')

    # Display the updated DataFrame to the user
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Category Mapping to Home Appliances", dataframe=df)


    # List of categories to be replaced with 'Others'
    categories_to_replace_others = [
    'Hair Dryers', 'Musical Instruments', 'Guitar', 
    'Furniture & Deco', 'Leisure & Transport', 
    'Food Processor', 'Showcase', 
    'No Category Found', 'Mini'
    ]

    # Replace the specified categories with 'Others'
    df['PreferredOrderCat'] = df['PreferredOrderCat'].replace(categories_to_replace_others, 'Others')

    # Display the updated DataFrame to the user
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Category Mapping to Others", dataframe=df)

    # List of categories to be replaced with 'Others'
    categories_to_replace_others = [
    'Hair Dryers', 'Musical Instruments', 'Guitar', 
    'Furniture & Deco', 'Leisure & Transport', 
    'Food Processor', 'Showcase', 
    'No Category Found', 'Mini'
    ]

    # Replace the specified categories with 'Others'
    df['PreferredOrderCat'] = df['PreferredOrderCat'].replace(categories_to_replace_others, 'Others')

    # Display the updated DataFrame to the user
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Category Mapping to Others", dataframe=df)

    tenure_data = df['Tenure']

    # Quartiles
    Q1 = tenure_data.quantile(0.25)
    median = tenure_data.median()
    Q3 = tenure_data.quantile(0.75)

    # Interquartile range
    IQR = Q3 - Q1

    # Whiskers
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR

    # print('Lower wishker',lower_whisker)
    # print('upper wishker',upper_range)

    # Outliers
    outliers = tenure_data[(tenure_data < lower_whisker) | (tenure_data > upper_whisker)]

    tenure_data = df['NumberOfAddress']

    # Quartiles
    Q1 = tenure_data.quantile(0.25)
    median = tenure_data.median()
    Q3 = tenure_data.quantile(0.75)

    # Interquartile range
    IQR = Q3 - Q1

    # Whiskers
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    df['NumberOfAddress']=np.where(
    df['NumberOfAddress']>5,
    5,
    np.where(df['NumberOfAddress']< lower_whisker,
             lower_whisker,df['NumberOfAddress']))


    tenure_data = df['OrderAmountHikeFromLastYear']

    # Quartiles
    Q1 = tenure_data.quantile(0.25)
    median = tenure_data.median()
    Q3 = tenure_data.quantile(0.75)

    # Interquartile range
    IQR = Q3 - Q1

    # Whiskers
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR

    df['OrderAmountHikeFromLastYear']=np.where(
    df['OrderAmountHikeFromLastYear']>upper_whisker,
    upper_whisker,
    np.where(df['OrderAmountHikeFromLastYear']< lower_whisker,
             lower_whisker,df['OrderAmountHikeFromLastYear']))

    tenure_data = df['DaySinceLastOrder']

    # Quartiles
    Q1 = tenure_data.quantile(0.25)
    median = tenure_data.median()
    Q3 = tenure_data.quantile(0.75)

    # Interquartile range
    IQR = Q3 - Q1

    # Whiskers
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    df['DaySinceLastOrder']=np.where(
    df['DaySinceLastOrder']>upper_whisker,
    upper_whisker,
    np.where(df['DaySinceLastOrder']< lower_whisker,
             lower_whisker,df['DaySinceLastOrder']))
    return df


# store the processed data locally 

def save_data(data_path,df):
    os.makedirs(data_path)
    df.to_csv(os.path.join(data_path,"dfprocessed_.csv"))


def main():
    df = data_preprocessed(df_unprocessed)
    data_path = os.path.join("data","interim")
    save_data(data_path,df)

if __name__ == "__main__":
    main()    

