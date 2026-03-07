import matplotlib.pyplot as plt
import seaborn as sns


def sales_trend(df):
    weekly_sales = df.groupby("Date")["Weekly_Sales"].sum()

    plt.figure(figsize=(12,6))
    plt.plot(weekly_sales)
    plt.title("Weekly Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()


def holiday_analysis(df):
    holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].mean()

    sns.barplot(x=holiday_sales.index, y=holiday_sales.values)
    plt.title("Holiday vs Non-Holiday Sales")
    plt.show()